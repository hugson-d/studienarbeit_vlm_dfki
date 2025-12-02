#!/usr/bin/env python3
"""
VLM Benchmark für Känguru-Mathematik-Aufgaben
Modell: Qwen2.5-VL-3B (Optimized: Batch Size 4 + Concise CoT)
"""

import os
import json
import torch
import logging
import re
import time
import random
import gc
import pandas as pd
from PIL import Image
from typing import Dict, List, Literal, Union, Any
from pathlib import Path
from tqdm import tqdm

from pydantic import BaseModel, ValidationError, Field

# ============================================================================
# SETUP & KONFIGURATION
# ============================================================================

MODEL_NAME = "Qwen2.5-VL-3B-Batch4-Concise"
MODEL_HF_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
BATCH_SIZE = 4  # <-- HIER: Parallelisierung
USE_QUANTIZATION = False # Bei 3B und Batch 4 reicht 24GB VRAM meist auch ohne Quant.
                         # Setzen Sie True, falls Sie OOM Fehler bekommen.

_script_path = Path(__file__).resolve()
PROJECT_ROOT = Path(os.environ.get("VLM_PROJECT_ROOT", _script_path.parent.parent.parent))
DATASET_PATH = PROJECT_ROOT / "dataset_final.json"
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "evaluation_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = OUTPUT_DIR / f"{MODEL_NAME}_results.jsonl"
EXCEL_FILE = OUTPUT_DIR / f"{MODEL_NAME}_summary.xlsx"
SEED = 42

# HuggingFace Login
from huggingface_hub import login
HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN:
    login(token=HF_TOKEN)

from transformers import (
    AutoProcessor, 
    Qwen2_5_VLForConditionalGeneration,
    BitsAndBytesConfig
)
from qwen_vl_utils import process_vision_info

# Logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler(OUTPUT_DIR / f"{MODEL_NAME}.log")]
)
logger = logging.getLogger(MODEL_NAME)

# ============================================================================
# PARSING SCHEMA
# ============================================================================

class KanguruAnswer(BaseModel):
    answer: Literal['A', 'B', 'C', 'D', 'E'] = Field(description="Die korrekte Antwort.")

def parse_response(output_text: str) -> Dict:
    clean_text = output_text.strip()
    # Markdown Code entfernen
    if "```" in clean_text:
        match = re.search(r'```(?:json)?\s*(.*?)\s*```', clean_text, re.DOTALL)
        if match: clean_text = match.group(1).strip()
    
    # JSON Suche
    json_match = re.search(r'\{[^{}]*\}', clean_text, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group(0))
            validated = KanguruAnswer(**data)
            return {"prediction": validated.answer, "format_valid": True, "error": None}
        except Exception: pass

    # Fallback Regex
    patterns = [r'(?:antwort|answer)[:\s]+([A-E])\b', r'"answer"\s*:\s*"?([A-E])"?']
    for p in patterns:
        m = re.search(p, clean_text, re.IGNORECASE)
        if m: return {"prediction": m.group(1).upper(), "format_valid": False, "error": "Regex Extraction"}
    
    return {"prediction": None, "format_valid": False, "error": "No valid answer"}

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def free_gpu_memory():
    gc.collect()
    torch.cuda.empty_cache()

# ============================================================================
# EVALUATOR MIT BATCH SUPPORT
# ============================================================================

class VLMEvaluator:
    def __init__(self):
        logger.info(f"🏗️ Lade {MODEL_NAME} | Batch Size: {BATCH_SIZE}")
        
        bnb_config = None
        if USE_QUANTIZATION:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", 
                bnb_4bit_compute_dtype=torch.bfloat16
            )

        self.processor = AutoProcessor.from_pretrained(MODEL_HF_ID, min_pixels=256*28*28, max_pixels=1280*28*28)
        
        load_kwargs = {"device_map": "auto", "torch_dtype": torch.bfloat16}
        if bnb_config: load_kwargs["quantization_config"] = bnb_config
        
        try:
            import flash_attn
            load_kwargs["attn_implementation"] = "flash_attention_2"
            logger.info("   ⚡ Flash Attention 2 aktiviert")
        except: pass

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(MODEL_HF_ID, **load_kwargs).eval()

    def generate_batch(self, tasks: List[Dict]) -> List[Dict]:
        """Verarbeitet eine Liste von Aufgaben gleichzeitig."""
        
        # 1. Bilder laden und Messages vorbereiten
        batch_messages = []
        valid_indices = []
        
        # CONCISE CoT PROMPT (Optimiert für Speed & Token-Ersparnis)
        system_prompt = (
            "Du bist ein Mathe-Profi. Löse die Aufgabe effizient.\n"
            "REGELN:\n"
            "1. Denke Schritt für Schritt, aber nutze NUR Stichpunkte.\n"
            "2. Verwende Formeln statt Text wo möglich.\n"
            "3. Fasse dich extrem kurz.\n"
            "4. Am Ende MUSS ein JSON stehen: {\"answer\": \"X\"}"
        )
        user_prompt = "Löse die Aufgabe. Kurzfassung. JSON am Ende."

        for i, task in enumerate(tasks):
            full_path = DATA_DIR / task["image_path"]
            if not full_path.exists():
                logger.warning(f"Bild fehlt: {full_path}")
                continue
            
            try:
                image = Image.open(full_path).convert("RGB")
                msg = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": user_prompt}
                    ]}
                ]
                batch_messages.append(msg)
                valid_indices.append(i)
            except Exception as e:
                logger.error(f"Ladefehler bei {task.get('task_id')}: {e}")

        if not batch_messages:
            return [{"error": "No valid images"}] * len(tasks)

        # 2. Batch Processing mit Qwen Utils
        try:
            # Prepare Inputs
            text_prompts = [
                self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) 
                for msg in batch_messages
            ]
            
            # process_vision_info extrahiert Bilder aus der Message-Liste
            image_inputs, video_inputs = process_vision_info(batch_messages)
            
            inputs = self.processor(
                text=text_prompts,
                images=image_inputs,
                videos=video_inputs,
                padding=True, # WICHTIG für Batching
                return_tensors="pt"
            ).to(self.model.device)

            # 3. Generierung
            start_time = time.time()
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=512, # Weniger Tokens nötig dank Concise CoT
                    do_sample=False,    # Greedy decoding
                )
            duration = time.time() - start_time
            avg_time_per_item = duration / len(batch_messages)

            # 4. Decoding
            input_len = inputs.input_ids.shape[1]
            # Trimmen: Qwen gibt Input+Output zurück, wir wollen nur Output
            trimmed_ids = generated_ids[:, input_len:]
            output_texts = self.processor.batch_decode(trimmed_ids, skip_special_tokens=True)

            # 5. Ergebnisse mappen
            results = [None] * len(tasks)
            
            # Fehlerfälle füllen
            for idx in range(len(tasks)):
                if idx not in valid_indices:
                    results[idx] = {"error": "Image load failed", "prediction": None}

            # Erfolgreiche Fälle füllen
            for batch_idx, original_idx in enumerate(valid_indices):
                raw_txt = output_texts[batch_idx]
                parsed = parse_response(raw_txt)
                results[original_idx] = {
                    "raw_output": raw_txt,
                    "prediction": parsed["prediction"],
                    "format_valid": parsed["format_valid"],
                    "error": parsed["error"],
                    "inference_time": round(avg_time_per_item, 4), # Zeit anteilig
                    "total_batch_time": round(duration, 4)
                }
            
            return results

        except Exception as e:
            logger.error(f"Batch Error: {e}")
            # Fallback: Alles als Error markieren
            return [{"error": f"Batch Failed: {str(e)}", "prediction": None}] * len(tasks)

    def cleanup(self):
        del self.model
        del self.processor
        free_gpu_memory()

# ============================================================================
# MAIN
# ============================================================================

def chunked_iterable(iterable, size):
    """Hilfsfunktion um Listen in Chunks zu teilen."""
    for i in range(0, len(iterable), size):
        yield iterable[i:i + size]

def run_benchmark():
    set_seed(SEED)
    
    if not DATASET_PATH.exists(): return
    with open(DATASET_PATH) as f: dataset = json.load(f)

    # Bereits bearbeitete filtern
    processed_ids = set()
    if LOG_FILE.exists():
        with open(LOG_FILE, 'r') as f:
            for line in f:
                try: processed_ids.add(json.loads(line)['task_id']) 
                except: pass

    # Aufgaben vorbereiten (IDs generieren)
    tasks_to_process = []
    for t in dataset:
        tid = f"{t.get('year')}_{t.get('class')}_{t.get('task_id')}"
        if tid not in processed_ids:
            t["full_id"] = tid
            tasks_to_process.append(t)

    if not tasks_to_process:
        logger.info("Alles erledigt.")
        return

    evaluator = VLMEvaluator()
    
    # BATCH LOOP
    correct_count = 0
    processed_count = 0
    
    with open(LOG_FILE, 'a') as f_log:
        # TQDM zeigt nun Batches an
        pbar = tqdm(chunked_iterable(tasks_to_process, BATCH_SIZE), 
                   total=(len(tasks_to_process) + BATCH_SIZE - 1) // BATCH_SIZE,
                   desc=f"{MODEL_NAME}")
        
        for batch_tasks in pbar:
            batch_results = evaluator.generate_batch(batch_tasks)
            
            for task, res in zip(batch_tasks, batch_results):
                gt = task.get('answer')
                pred = res.get('prediction')
                is_correct = (pred == gt) if pred else False
                
                if is_correct: correct_count += 1
                processed_count += 1

                log_entry = {
                    "task_id": task["full_id"],
                    "year": task.get("year"),
                    "class": task.get("class"),
                    "math_category": task.get("math_category"),
                    "ground_truth": gt,
                    "prediction": pred,
                    "is_correct": is_correct,
                    "format_valid": res.get("format_valid"),
                    "raw_output": res.get("raw_output"),
                    "inference_time": res.get("inference_time")
                }
                f_log.write(json.dumps(log_entry) + "\n")
            
            f_log.flush()
            acc = correct_count / processed_count if processed_count > 0 else 0
            pbar.set_postfix({"acc": f"{acc:.1%}"})

    evaluator.cleanup()
    generate_report()

def generate_report():
    if not LOG_FILE.exists(): return
    df = pd.read_json(LOG_FILE, lines=True)
    if df.empty: return
    df.to_excel(EXCEL_FILE, index=False)
    print(f"\n📊 {MODEL_NAME} | Acc: {df['is_correct'].mean():.1%} | Avg Time: {df['inference_time'].mean():.2f}s")

if __name__ == "__main__":
    run_benchmark()