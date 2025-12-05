#!/usr/bin/env python3
import os
import json
import torch
import logging
import re
import time
import argparse
import pandas as pd
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM

# --- KONFIGURATION ---
MODEL_ID = "AIDC-AI/Ovis2.5-9B"
MODEL_NAME = "Ovis2.5-9B"

# Pfade setup
_script_path = Path(__file__).resolve()
PROJECT_ROOT = Path(os.environ.get("VLM_PROJECT_ROOT", _script_path.parent.parent.parent))
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "evaluation_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = OUTPUT_DIR / f"{MODEL_NAME}_results.jsonl"

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger()

def setup_model():
    """
    Lädt das Modell exakt wie im Colab-Snippet, aber mit device_map für Cluster-Sicherheit.
    """
    logger.info(f"Lade Modell: {MODEL_ID} ...")
    
    # Colab-Logik: trust_remote_code=True ist entscheidend
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16, # BF16 ist auf A100 besser/schneller als "auto"
        device_map="auto",          # Automatische GPU-Verteilung (verhindert OOM beim Laden)
        low_cpu_mem_usage=True
    ).eval()
    
    # Tokenizer holen (Ovis spezifisch)
    if hasattr(model, "get_text_tokenizer"):
        tokenizer = model.get_text_tokenizer()
    elif hasattr(model, "text_tokenizer"):
        tokenizer = model.text_tokenizer
    else:
        raise ValueError("Konnte Text-Tokenizer nicht finden.")
        
    logger.info("Modell erfolgreich geladen.")
    return model, tokenizer

def extract_answer(text):
    """Extrahiert A, B, C, D, E aus dem Output."""
    text = text.strip()
    # 1. Suche nach JSON
    import json
    try:
        # Suche nach dem letzten JSON Block
        matches = re.findall(r"\{.*?\}", text, re.DOTALL)
        if matches:
            data = json.loads(matches[-1])
            if "answer" in data:
                return data["answer"], True
    except:
        pass
    
    # 2. Fallback Regex
    match = re.search(r"Answer:\s*([A-E])", text, re.IGNORECASE)
    if match: return match.group(1).upper(), False
    
    # 3. Nur Buchstabe
    if text in ["A", "B", "C", "D", "E"]: return text, False
    
    return None, False

def main():
    # 1. Daten laden
    dataset_path = PROJECT_ROOT / "dataset_final.json"
    if not dataset_path.exists():
        logger.error(f"Kein Dataset gefunden unter {dataset_path}")
        return
        
    with open(dataset_path) as f:
        dataset = json.load(f)

    # 2. Bereits bearbeitete filtern
    processed = set()
    if LOG_FILE.exists():
        with open(LOG_FILE) as f:
            for line in f:
                try: processed.add(json.loads(line)["task_id"])
                except: pass

    # 3. Modell laden
    model, tokenizer = setup_model()

    # 4. Loop
    with open(LOG_FILE, "a") as f_out:
        for task in tqdm(dataset, desc="Eval"):
            t_id = f"{task.get('year')}_{task.get('class')}_{task.get('task_id')}"
            if t_id in processed: continue

            img_path = DATA_DIR / task["image_path"]
            if not img_path.exists(): continue

            try:
                image = Image.open(img_path).convert("RGB")
                
                # Prompting Strategie
                prompt = "Analyze the image and solve the math problem. Output the final answer as a JSON object: {\"answer\": \"X\"} where X is A, B, C, D, or E."
                
                messages = [{
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt}
                    ]
                }]

                # Ovis Preprocessing
                inputs, pixel_values, grid_thws = model.preprocess_inputs(
                    messages=messages,
                    add_generation_prompt=True
                )

                # Move to device
                inputs = inputs.to(model.device)
                pixel_values = pixel_values.to(model.device, dtype=model.dtype) if pixel_values is not None else None
                grid_thws = grid_thws.to(model.device) if grid_thws is not None else None

                # Generate
                with torch.inference_mode():
                    outputs = model.generate(
                        inputs=inputs,
                        pixel_values=pixel_values,
                        grid_thws=grid_thws,
                        max_new_tokens=128,
                        do_sample=False,
                        eos_token_id=tokenizer.eos_token_id,
                        pad_token_id=tokenizer.pad_token_id
                    )

                # Decode
                response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                pred, valid_fmt = extract_answer(response)
                
                # Save
                res = {
                    "task_id": t_id,
                    "prediction": pred,
                    "ground_truth": task["answer"],
                    "is_correct": pred == task["answer"],
                    "raw_output": response
                }
                f_out.write(json.dumps(res) + "\n")
                f_out.flush()

            except Exception as e:
                logger.error(f"Error Task {t_id}: {e}")

if __name__ == "__main__":
    main()