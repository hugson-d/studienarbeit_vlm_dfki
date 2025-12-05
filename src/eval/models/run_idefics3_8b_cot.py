#!/usr/bin/env python3
"""
VLM Benchmark für Känguru-Mathematik-Aufgaben
Modell: HuggingFaceM4/Idefics3-8B-Llama3
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
from typing import Dict, List, Literal, Union
from pathlib import Path
from tqdm import tqdm
from pydantic import BaseModel, ValidationError, Field

# Projekt-Root
_script_path = Path(__file__).resolve()
PROJECT_ROOT = Path(os.environ.get("VLM_PROJECT_ROOT", _script_path.parent.parent.parent))
DATASET_PATH = PROJECT_ROOT / "dataset_final.json"
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "evaluation_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# HuggingFace Login
from huggingface_hub import login
HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN:
    login(token=HF_TOKEN)

from transformers import (
    AutoProcessor,
    AutoModelForVision2Seq
)

# ============================================================================
# KONFIGURATION - IDEFICS3
# ============================================================================

MODEL_NAME = "Idefics3-8B-Llama3-CoT"
MODEL_HF_ID = "HuggingFaceM4/Idefics3-8B-Llama3"
MODEL_PARAMS_B = 8

SEED = 42
LOG_FILE = OUTPUT_DIR / f"{MODEL_NAME}_results.jsonl"
EXCEL_FILE = OUTPUT_DIR / f"{MODEL_NAME}_summary.xlsx"

# Logging Setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(OUTPUT_DIR / f"{MODEL_NAME}.log")
    ]
)
logger = logging.getLogger(MODEL_NAME)

# Reduziere transformers Verbosity
import transformers
transformers.logging.set_verbosity_error()

# ============================================================================
# UTILS & PARSING (Identisch zu anderen Modellen für Vergleichbarkeit)
# ============================================================================

class KanguruAnswer(BaseModel):
    answer: Literal['A', 'B', 'C', 'D', 'E'] = Field(description="Die korrekte Antwort.")

def parse_response(output_text: str) -> Dict:
    clean_text = output_text.strip()
    # Markdown Code-Block entfernen
    if "```" in clean_text:
        match = re.search(r'```(?:json)?\s*(.*?)\s*```', clean_text, re.DOTALL)
        if match:
            clean_text = match.group(1).strip()

    # JSON Suche
    json_match = re.search(r'\{[^{}]*\}', clean_text, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group(0))
            data = {k.lower(): v for k, v in data.items()}
            validated = KanguruAnswer(**data)
            return {"prediction": validated.answer, "format_valid": True, "error": None}
        except Exception:
            pass

    # Regex Fallback
    patterns = [
        r'(?:antwort|answer|lösung|solution)[:\s]+([A-E])\b',
        r'\b([A-E])\s*(?:ist|is)\s+(?:richtig|correct)',
        r'"answer"\s*:\s*"?([A-E])"?'
    ]
    for p in patterns:
        m = re.search(p, clean_text, re.IGNORECASE)
        if m:
            return {"prediction": m.group(1).upper(), "format_valid": False, "error": "Regex Extraction"}

    # Letzter Fallback: Suche nach dem letzten Vorkommen eines Buchstabens A-E
    last_letter_match = re.findall(r'\b([A-E])\b', clean_text.upper())
    if last_letter_match:
        return {"prediction": last_letter_match[-1], "format_valid": False, "error": "Fallback: Last A-E"}

    return {"prediction": None, "format_valid": False, "error": "No valid answer"}

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def free_gpu_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# ============================================================================
# EVALUATOR
# ============================================================================

class VLMEvaluator:
    def __init__(self):
        logger.info(f"🏗️ Lade {MODEL_NAME} ({MODEL_PARAMS_B}B)")

        # 1. Processor (use_fast=True reduziert Token-Output)
        self.processor = AutoProcessor.from_pretrained(
            MODEL_HF_ID,
            use_fast=True
        )

        # 2. Modell (offizielle HF API)
        self.model = AutoModelForVision2Seq.from_pretrained(
            MODEL_HF_ID,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        ).eval()

        logger.info(f"✅ {MODEL_NAME} bereit auf {self.model.device}")

    def generate(self, image_path: str) -> Dict:
        full_path = DATA_DIR / image_path
        if not full_path.exists():
            logger.error(f"❌ Bild nicht gefunden: {full_path}")
            logger.error(f"   DATA_DIR: {DATA_DIR}")
            logger.error(f"   image_path: {image_path}")
            return {"error": f"Image not found: {full_path}", "prediction": None}

        # Bild laden
        image = Image.open(full_path).convert("RGB")

        # CHAIN-OF-THOUGHT PROMPT (Exakt wie Ovis/InternVL CoT)
        system_prompt = (
            "Du bist ein mathematisches Assistenzsystem für Multiple-Choice-Aufgaben.\n\n"
            "ARBEITSWEISE:\n"
            "Denke gründlich über die Aufgabe nach und analysiere sie intern Schritt für Schritt.\n\n"
            "ZWINGENDE AUSGABE - NUR DIESES FORMAT IST ERLAUBT:\n"
            '{"answer": "X"}\n'
            "wobei X EXAKT einer dieser Buchstaben sein MUSS: A, B, C, D oder E\n\n"
            "WICHTIG:\n"
            "- Deine GESAMTE Antwort besteht NUR aus diesem JSON-Objekt.\n"
            "- KEINE Erklärungen, Rechenwege oder Zwischenschritte in der Ausgabe.\n"
            "- Denke intern, aber gib NUR das JSON aus.\n"
            "- Bei Unsicherheit: Wähle die wahrscheinlichste Option (A-E).\n"
            "- Eine Antwort ist PFLICHT - du musst A, B, C, D oder E wählen."
        )
        user_prompt = "Denke gründlich nach und gib dann NUR das JSON mit deiner Antwort zurück."

        # Chat Template Erstellung (offizielle HF API)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": f"{system_prompt}\n\n{user_prompt}"}
                ]
            }
        ]

        # apply_chat_template für Prompt-Formatierung (offizielles HF Format)
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        
        # Processor mit Text UND Bildern (wie in HF Beispiel)
        inputs = self.processor(text=prompt, images=[image], return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # Generierung
        start_time = time.time()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
            )
        duration = time.time() - start_time

        # Nur neue Tokens decodieren (Prompt entfernen)
        input_len = inputs["input_ids"].shape[-1]
        output_text = self.processor.decode(
            outputs[0][input_len:],
            skip_special_tokens=True
        )

        # Parsing
        result = parse_response(output_text)

        return {
            "prediction": result["prediction"],
            "format_valid": result["format_valid"],
            "error": result["error"],
            "inference_time": round(duration, 4),
            "input_tokens": input_len
        }

    def cleanup(self):
        del self.model
        del self.processor
        free_gpu_memory()

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def run_benchmark():
    set_seed(SEED)

    if not DATASET_PATH.exists():
        logger.error(f"Dataset fehlt: {DATASET_PATH}")
        return

    with open(DATASET_PATH) as f:
        dataset = json.load(f)

    # Resume-Logik
    processed_ids = set()
    if LOG_FILE.exists():
        with open(LOG_FILE, 'r') as f:
            for line in f:
                try:
                    processed_ids.add(json.loads(line)['task_id'])
                except Exception:
                    pass

    evaluator = VLMEvaluator()

    correct_count = 0
    processed_count = 0

    with open(LOG_FILE, 'a') as f_log:
        pbar = tqdm(dataset, desc=MODEL_NAME)
        for task in pbar:
            task_id = f"{task.get('year')}_{task.get('class')}_{task.get('task_id')}"

            if task_id in processed_ids:
                continue

            try:
                result = evaluator.generate(task.get("image_path"))

                gt = task.get('answer')
                is_correct = (result['prediction'] == gt) if result['prediction'] else False

                if is_correct:
                    correct_count += 1
                processed_count += 1

                log_entry = {
                    "model": MODEL_NAME,
                    "task_id": task_id,
                    "year": task.get("year"),
                    "class": task.get("class"),
                    "original_task_id": task.get("task_id"),
                    "math_category": task.get("math_category"),
                    "is_text_only": task.get("is_text_only"),
                    "ground_truth": gt,
                    "prediction": result["prediction"],
                    "is_correct": is_correct,
                    "format_valid": result.get("format_valid"),
                    "error_type": result.get("error"),
                    "inference_time": result.get("inference_time"),
                    "input_tokens": result.get("input_tokens")
                }

                f_log.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
                f_log.flush()

                acc = correct_count / processed_count if processed_count > 0 else 0
                pbar.set_postfix({"acc": f"{acc:.1%}"})

            except Exception as e:
                logger.error(f"Fehler bei {task_id}: {e}")
                if "out of memory" in str(e).lower():
                    logger.critical("OOM Error! Abbruch.")
                    break

    evaluator.cleanup()
    generate_report()

def generate_report():
    if not LOG_FILE.exists():
        return
    df = pd.read_json(LOG_FILE, lines=True)
    if df.empty:
        return

    df.to_excel(EXCEL_FILE, index=False)

    print("\n" + "=" * 70)
    print(f"📊 ERGEBNISSE: {MODEL_NAME}")
    print(f"  Accuracy:     {df['is_correct'].mean():.1%}")
    if 'format_valid' in df.columns:
        print(f"  Valid JSON:   {df['format_valid'].mean():.1%}")

    if 'math_category' in df.columns:
        print("\n📐 Nach Kategorie:")
        print(df.groupby('math_category')['is_correct'].mean().apply(lambda x: f"{x:.1%}"))

if __name__ == "__main__":
    run_benchmark()
