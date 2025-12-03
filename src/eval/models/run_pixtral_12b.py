#!/usr/bin/env python3
"""
VLM Benchmark für Känguru-Mathematik-Aufgaben
Modell: mistralai/Pixtral-12B-2409 (Multimodal/VLM)
Inference: vLLM mit Mistral Tokenizer
"""

import os
import json
import logging
import re
import time
import random
import gc
import pandas as pd
from PIL import Image
from typing import Dict, List, Literal
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

# vLLM Import
from vllm import LLM, SamplingParams

# ============================================================================
# KONFIGURATION - PIXTRAL 12B
# ============================================================================

MODEL_NAME = "Pixtral-12B-2409"
MODEL_HF_ID = "mistralai/Pixtral-12B-2409"

# Pixtral: 12B Decoder + 400M Vision Encoder = ~12.4B Parameter
# BF16 benötigt ~25GB VRAM - passt auf A100-40GB/80GB

SEED = 42
LOG_FILE = OUTPUT_DIR / f"{MODEL_NAME}_results.jsonl"
EXCEL_FILE = OUTPUT_DIR / f"{MODEL_NAME}_summary.xlsx"

# vLLM Konfiguration
MAX_MODEL_LEN = 32768  # Reduziert für VRAM-Effizienz (Original: 128k)
MAX_TOKENS = 128  # Für JSON-Output ausreichend

# Logging Setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(OUTPUT_DIR / f"{MODEL_NAME}.log")
    ]
)
logger = logging.getLogger(MODEL_NAME)

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
        r'(?:antwort|answer|lösung|solution)[:\s]+([A-E])\b',
        r'\b([A-E])\s*(?:ist|is)\s+(?:richtig|correct)',
        r'"answer"\s*:\s*"?([A-E])"?'
    ]
    for p in patterns:
        m = re.search(p, clean_text, re.IGNORECASE)
        if m:
            return {"prediction": m.group(1).upper(), "format_valid": False, "error": "Regex Extraction"}

    return {"prediction": None, "format_valid": False, "error": "No valid answer"}

def set_seed(seed):
    random.seed(seed)

def free_gpu_memory():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass

# ============================================================================
# EVALUATOR (PIXTRAL 12B mit vLLM)
# ============================================================================

class VLMEvaluator:
    def __init__(self):
        logger.info(f"Lade {MODEL_NAME} ({MODEL_HF_ID}) mit vLLM")

        # vLLM LLM mit Mistral Tokenizer
        # WICHTIG: tokenizer_mode="mistral" ist erforderlich für Pixtral
        self.llm = LLM(
            model=MODEL_HF_ID,
            tokenizer_mode="mistral",  # Spezifisch für Pixtral!
            max_model_len=MAX_MODEL_LEN,
            limit_mm_per_prompt={"image": 1},  # Ein Bild pro Prompt
            trust_remote_code=True,
        )

        # Sampling Parameter (Greedy für Vergleichbarkeit)
        self.sampling_params = SamplingParams(
            max_tokens=MAX_TOKENS,
            temperature=0.0,  # Greedy decoding
        )

        logger.info(f"{MODEL_NAME} bereit")

    def generate(self, image_path: str) -> Dict:
        full_path = DATA_DIR / image_path
        if not full_path.exists():
            return {"error": "Image not found", "prediction": None}

        # PROMPT
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

        # vLLM Chat Format für Pixtral
        # Bild wird als file:// URL übergeben
        image_url = f"file://{full_path}"
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"{system_prompt}\n\n{user_prompt}"},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        ]

        start_time = time.time()
        
        # vLLM Chat Interface
        outputs = self.llm.chat(messages, sampling_params=self.sampling_params)
        
        duration = time.time() - start_time

        # Output extrahieren
        output_text = outputs[0].outputs[0].text

        # Parsing
        result = parse_response(output_text)

        return {
            "prediction": result["prediction"],
            "format_valid": result["format_valid"],
            "error": result["error"],
            "inference_time": round(duration, 4),
            "raw_output": output_text[:500],  # Für Debugging
        }

    def cleanup(self):
        del self.llm
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
        with open(LOG_FILE, "r") as f:
            for line in f:
                try:
                    processed_ids.add(json.loads(line)["task_id"])
                except:
                    pass

    evaluator = VLMEvaluator()

    correct_count = 0
    processed_count = 0

    with open(LOG_FILE, "a") as f_log:
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
    print(f"  Valid JSON:   {df['format_valid'].mean():.1%}")

    if "math_category" in df.columns:
        print("\n📐 Nach Kategorie:")
        print(df.groupby("math_category")["is_correct"].mean().apply(lambda x: f"{x:.1%}"))


if __name__ == "__main__":
    run_benchmark()
