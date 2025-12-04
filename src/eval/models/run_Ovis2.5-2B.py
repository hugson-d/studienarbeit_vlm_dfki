#!/usr/bin/env python3
"""
VLM Benchmark für Känguru-Mathematik-Aufgaben
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
from transformers import AutoModelForCausalLM

# ============================================================================
# ARGPARSE / KONFIG
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="VLM Benchmark für Känguru-Mathematik-Aufgaben (Ovis2 / Ovis2.5)"
    )
    parser.add_argument(
        "--hf-id",
        type=str,
        default="AIDC-AI/Ovis2.5-2B"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="Ovis2.5-2B",
        help="Interner Modellname für Logs/Dateien (frei wählbar).",
    )
    return parser.parse_args()

args = parse_args()
MODEL_HF_ID = args.hf_id
MODEL_NAME = args.model_name

# ============================================================================
# PFAD & LOGGING
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
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(OUTPUT_DIR / f"{MODEL_NAME}.log")
    ]
)
logger = logging.getLogger(MODEL_NAME)

# HuggingFace Login
HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN:
    login(token=HF_TOKEN)

# ============================================================================
# UTILS & PARSING
# ============================================================================

class KanguruAnswer(BaseModel):
    answer: Literal["A", "B", "C", "D", "E"] = Field(description="Die korrekte Antwort.")

def parse_response(output_text: str) -> Dict:
    clean_text = output_text.strip()
    # Markdown Code-Block entfernen
    if "```" in clean_text:
        match = re.search(r"```(?:json)?\s*(.*?)\s*```", clean_text, re.DOTALL)
        if match:
            clean_text = match.group(1).strip()

    # JSON Suche
    json_match = re.search(r"\{[^{}]*\}", clean_text, re.DOTALL)
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
        r"(?:antwort|answer|lösung|solution)[:\s]+([A-E])\b",
        r"\b([A-E])\s*(?:ist|is)\s+(?:richtig|correct)",
        r'"answer"\s*:\s*"?([A-E])"?'
    ]
    for p in patterns:
        m = re.search(p, clean_text, re.IGNORECASE)
        if m:
            return {
                "prediction": m.group(1).upper(),
                "format_valid": False,
                "error": "Regex Extraction",
            }

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
# EVALUATOR (OVIS2 / OVIS2.5)
# ============================================================================

class VLMEvaluator:
    def __init__(self):
        logger.info(f"Lade {MODEL_NAME} ({MODEL_HF_ID})")

        load_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.bfloat16,
            "low_cpu_mem_usage": True,
        }

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_HF_ID,
            **load_kwargs,
        ).to(device).eval()

        # Ovis text tokenizer
        # Ovis2.5 nutzt teilweise get_text_tokenizer(); Ovis2-Modelle stellen oft text_tokenizer bereit.
        # Wir versuchen erst text_tokenizer, dann Fallback auf get_text_tokenizer.
        if hasattr(self.model, "text_tokenizer"):
            self.tokenizer = self.model.text_tokenizer
        elif hasattr(self.model, "get_text_tokenizer"):
            self.tokenizer = self.model.get_text_tokenizer()
        else:
            raise AttributeError("Modell bietet keinen text_tokenizer / get_text_tokenizer an.")

        logger.info(f"{MODEL_NAME} bereit auf {self.model.device}")

    def generate(self, image_path: str) -> Dict:
        full_path = DATA_DIR / image_path
        if not full_path.exists():
            return {"error": "Image not found", "prediction": None}

        image = Image.open(full_path).convert("RGB")

        system_prompt = (
            "Du bist ein mathematisches Assistenzsystem für Multiple-Choice-Aufgaben.\n\n"
            "AUFGABE: Analysiere das Bild und wähle die korrekte Antwort.\n\n"
            "ZWINGENDE AUSGABE - NUR DIESES FORMAT IST ERLAUBT:\n"
            '{"answer": "X"}\n'
            "wobei X EXAKT einer dieser Buchstaben sein MUSS: A, B, C, D oder E\n\n"
            "WICHTIG:\n"
            "- Deine GESAMTE Antwort besteht NUR aus diesem JSON-Objekt.\n"
            "- KEINE anderen Zeichen, Wörter oder Erklärungen.\n"
            "- Bei Unsicherheit: Wähle die wahrscheinlichste Option (A-E).\n"
            "- Eine Antwort ist PFLICHT - du musst A, B, C, D oder E wählen."
        )
        user_prompt = "Bestimme die korrekte Antwort basierend auf dem Bild. Gib nur das JSON zurück."

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": f"{system_prompt}\n\n{user_prompt}"},
                ],
            }
        ]

        # Ovis2 / Ovis2.5 preprocess_inputs
        input_ids, pixel_values, grid_thws = self.model.preprocess_inputs(
            messages=messages,
            add_generation_prompt=True,
        )

        device = self.model.device
        input_ids = input_ids.to(device)
        if pixel_values is not None:
            pixel_values = pixel_values.to(device, dtype=self.model.dtype)
        if grid_thws is not None:
            grid_thws = grid_thws.to(device)

        gen_kwargs = {
            "inputs": input_ids,
            "pixel_values": pixel_values,
            "grid_thws": grid_thws,
            "max_new_tokens": 512,
            "do_sample": False,
            "temperature": 0.0,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
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
        except Exception:
            pass
        try:
            del self.tokenizer
        except Exception:
            pass
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

    processed_ids = set()
    if LOG_FILE.exists():
        with open(LOG_FILE, "r") as f:
            for line in f:
                try:
                    processed_ids.add(json.loads(line)["task_id"])
                except Exception:
                    pass

    evaluator = VLMEvaluator()

    correct_count = 0
    processed_count = 0

    with open(LOG_FILE, "a", encoding="utf-8") as f_log:
        pbar = tqdm(dataset, desc=MODEL_NAME)
        for task in pbar:
            task_id = f"{task.get('year')}_{task.get('class')}_{task.get('task_id')}"

            if task_id in processed_ids:
                continue

            try:
                result = evaluator.generate(task.get("image_path"))

                gt = task.get("answer")
                is_correct = (result["prediction"] == gt) if result["prediction"] else False

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
                    "input_tokens": result.get("input_tokens"),
                }

                f_log.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
                f_log.flush()

                acc = correct_count / processed_count if processed_count > 0 else 0.0
                pbar.set_postfix({"acc": f"{acc:.1%}"})

            except Exception as e:
                logger.error(f"Fehler bei {task_id}: {e}")
                if "out of memory" in str(e).lower():
                    logger.critical("OOM Error, Abbruch.")
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
    print(f"ERGEBNISSE: {MODEL_NAME}")
    print(f"  Accuracy:     {df['is_correct'].mean():.1%}")
    print(f"  Valid JSON:   {df['format_valid'].mean():.1%}")

if __name__ == "__main__":
    run_benchmark()
