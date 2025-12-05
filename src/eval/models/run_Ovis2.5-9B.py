#!/usr/bin/env python3
"""
VLM Benchmark für Känguru-Mathematik-Aufgaben (Ovis2.5-9B Version)
Final Fix: Cache-Robustheit + Ovis-Specific Arguments
"""

import os
import json
import torch
import logging
import re
import time
import random
import gc
import argparse
import pandas as pd
from PIL import Image
from typing import Dict, Literal
from pathlib import Path
from tqdm import tqdm
from pydantic import BaseModel, Field
from huggingface_hub import login
# WICHTIG: AutoConfig explizit importieren für den Fix
from transformers import AutoModelForCausalLM, AutoConfig

# ============================================================================
# ARGPARSE
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="VLM Benchmark Ovis2.5-9B")
    parser.add_argument("--hf-id", type=str, default="AIDC-AI/Ovis2.5-9B")
    parser.add_argument("--model-name", type=str, default="Ovis2.5-9B")
    # Optional: Thinking Mode per Flag aktivieren
    parser.add_argument("--enable-thinking", action="store_true", help="Aktiviert Chain-of-Thought (langsamer, evtl. besser)")
    return parser.parse_args()

args = parse_args()
MODEL_HF_ID = args.hf_id
MODEL_NAME = args.model_name
ENABLE_THINKING = args.enable_thinking

# ============================================================================
# PFADE & LOGGING
# ============================================================================

_script_path = Path(__file__).resolve()
PROJECT_ROOT = Path(os.environ.get("VLM_PROJECT_ROOT", _script_path.parent.parent.parent))
DATASET_PATH = PROJECT_ROOT / "dataset_final.json"
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "evaluation_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
LOG_FILE = OUTPUT_DIR / f"{MODEL_NAME}_results.jsonl"
EXCEL_FILE = OUTPUT_DIR / f"{MODEL_NAME}_summary.xlsx"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler(OUTPUT_DIR / f"{MODEL_NAME}.log")]
)
logger = logging.getLogger(MODEL_NAME)

HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN:
    login(token=HF_TOKEN)

# ============================================================================
# UTILS
# ============================================================================

class KanguruAnswer(BaseModel):
    answer: Literal["A", "B", "C", "D", "E"] = Field(description="Die korrekte Antwort.")

def parse_response(output_text: str) -> Dict:
    clean_text = output_text.strip()
    
    # Falls Thinking Mode aktiv war, könnte viel Text vor dem JSON stehen.
    # Wir suchen das letzte Vorkommen von JSON-Codeblöcken oder geschweiften Klammern.
    
    # 1. Versuch: Markdown Code Block
    if "```" in clean_text:
        # Finde ALLE Codeblöcke und nimm den letzten (falls Thinking auch Code enthält)
        matches = re.findall(r"```(?:json)?\s*(.*?)\s*```", clean_text, re.DOTALL)
        if matches:
            clean_text = matches[-1].strip()

    # 2. Versuch: JSON Regex
    # Suche nach dem letzten JSON-Objekt im Text
    json_matches = re.findall(r"\{[^{}]*\}", clean_text, re.DOTALL)
    if json_matches:
        try:
            # Nimm das letzte gefundene Objekt (die Antwort kommt meist am Schluss)
            data = json.loads(json_matches[-1])
            data = {k.lower(): v for k, v in data.items()}
            validated = KanguruAnswer(**data)
            return {"prediction": validated.answer, "format_valid": True, "error": None}
        except Exception:
            pass

    # 3. Versuch: Regex Fallback
    patterns = [
        r"(?:antwort|answer|lösung|solution)[:\s]+([A-E])\b",
        r"\b([A-E])\s*(?:ist|is)\s+(?:richtig|correct)",
        r'"answer"\s*:\s*"?([A-E])"?'
    ]
    for p in patterns:
        m = re.search(p, clean_text, re.IGNORECASE)
        if m:
            return {"prediction": m.group(1).upper(), "format_valid": False, "error": "Regex Extraction"}

    return {"prediction": None, "format_valid": False, "error": "No valid answer"}

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def free_gpu_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# ============================================================================
# EVALUATOR (OVIS2.5-9B ROBUST)
# ============================================================================

class VLMEvaluator:
    def __init__(self):
        logger.info(f"Initialisiere {MODEL_NAME} ({MODEL_HF_ID})...")
        
        load_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.bfloat16,
            "low_cpu_mem_usage": True,
            "device_map": "auto",
        }

        # --- FIX FÜR DEN VALUE ERROR ---
        # Wir laden zuerst die Config, damit der 'ovis' Modell-Typ registriert wird,
        # bevor das schwere Modell geladen wird.
        logger.info("Schritt 1: Lade Config und registriere Remote Code...")
        try:
            config = AutoConfig.from_pretrained(MODEL_HF_ID, trust_remote_code=True)
        except Exception as e:
            logger.warning(f"Konnte Config nicht vorab laden ({e}). Versuche direkten Load.")
            config = None
        
        logger.info("Schritt 2: Lade Modell-Gewichte...")
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_HF_ID,
            config=config, # Wichtig!
            **load_kwargs,
        ).eval()

        if hasattr(self.model, "text_tokenizer"):
            self.tokenizer = self.model.text_tokenizer
        elif hasattr(self.model, "get_text_tokenizer"):
            self.tokenizer = self.model.get_text_tokenizer()
        else:
            raise AttributeError("Modell hat keinen text_tokenizer.")

        logger.info(f"{MODEL_NAME} erfolgreich geladen.")

    def generate(self, image_path: str) -> Dict:
        full_path = DATA_DIR / image_path
        if not full_path.exists():
            return {"error": "Image not found", "prediction": None}

        image = Image.open(full_path).convert("RGB")

        # Prompt Design
        system_prompt = (
            "Du bist ein mathematisches Assistenzsystem.\n"
            "Analysiere das Bild und wähle die korrekte Antwort (A, B, C, D, E).\n"
            "Gib am Ende NUR ein JSON zurück: {'answer': 'X'}."
        )
        user_prompt = "Löse die Aufgabe. Welche Antwort ist richtig?"

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": f"{system_prompt}\n\n{user_prompt}"},
                ],
            }
        ]

        # Parameter Setup basierend auf HuggingFace Snippet
        # Wir nutzen die Argumente, aber steuern sie über args (Default: False)
        input_ids, pixel_values, grid_thws = self.model.preprocess_inputs(
            messages=messages,
            add_generation_prompt=True,
            enable_thinking=ENABLE_THINKING, # HF Parameter
        )

        device = self.model.device
        input_ids = input_ids.to(device)
        if pixel_values is not None:
            pixel_values = pixel_values.to(device, dtype=self.model.dtype)
        if grid_thws is not None:
            grid_thws = grid_thws.to(device)

        # Budget Definition (nur relevant wenn Thinking=True)
        max_tokens = 3072 if ENABLE_THINKING else 512
        thinking_budget = 2048 if ENABLE_THINKING else 0

        gen_kwargs = {
            "inputs": input_ids,
            "pixel_values": pixel_values,
            "grid_thws": grid_thws,
            "max_new_tokens": max_tokens,
            "do_sample": False,
            "temperature": 0.0,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
            # Neue Ovis Parameter:
            "enable_thinking": ENABLE_THINKING,
            "enable_thinking_budget": ENABLE_THINKING,
            "thinking_budget": thinking_budget
        }

        start_time = time.time()
        with torch.inference_mode():
            outputs = self.model.generate(**gen_kwargs)
        duration = time.time() - start_time

        output_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        result = parse_response(output_text)

        return {
            "prediction": result["prediction"],
            "format_valid": result["format_valid"],
            "error": result["error"],
            "inference_time": round(duration, 4),
            "input_tokens": int(input_ids.shape[1]),
        }

    def cleanup(self):
        try:
            del self.model
            del self.tokenizer
        except:
            pass
        free_gpu_memory()

# ============================================================================
# MAIN
# ============================================================================

def run_benchmark():
    set_seed(SEED)
    if not DATASET_PATH.exists():
        logger.error(f"Dataset fehlt: {DATASET_PATH}")
        return

    with open(DATASET_PATH) as f:
        dataset = json.load(f)

    processed_ids = set()
    if LOG_FILE.exists():
        with open(LOG_FILE, "r") as f:
            for line in f:
                try:
                    processed_ids.add(json.loads(line)["task_id"])
                except: pass

    evaluator = VLMEvaluator()
    correct_count = 0
    processed_count = 0

    with open(LOG_FILE, "a", encoding="utf-8") as f_log:
        pbar = tqdm(dataset, desc=MODEL_NAME)
        for task in pbar:
            task_id = f"{task.get('year')}_{task.get('class')}_{task.get('task_id')}"
            if task_id in processed_ids: continue

            try:
                result = evaluator.generate(task.get("image_path"))
                gt = task.get("answer")
                is_correct = (result["prediction"] == gt) if result["prediction"] else False

                if is_correct: correct_count += 1
                processed_count += 1

                log_entry = {
                    "model": MODEL_NAME,
                    "task_id": task_id,
                    "year": task.get("year"),
                    "class": task.get("class"),
                    "prediction": result["prediction"],
                    "ground_truth": gt,
                    "is_correct": is_correct,
                    "format_valid": result["format_valid"],
                    "inference_time": result["inference_time"]
                }
                f_log.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
                f_log.flush()
                
                acc = correct_count / processed_count if processed_count > 0 else 0.0
                pbar.set_postfix({"acc": f"{acc:.1%}"})

            except Exception as e:
                logger.error(f"Fehler Task {task_id}: {e}")
                if "out of memory" in str(e).lower(): break

    evaluator.cleanup()
    
    if LOG_FILE.exists():
        df = pd.read_json(LOG_FILE, lines=True)
        df.to_excel(EXCEL_FILE, index=False)
        print(f"Accuracy: {df['is_correct'].mean():.1%}")

if __name__ == "__main__":
    run_benchmark()