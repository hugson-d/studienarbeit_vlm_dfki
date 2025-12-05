#!/usr/bin/env python3
"""
VLM Benchmark für Känguru-Mathematik-Aufgaben
Modell: mistralai/Mistral-Small-3.2-24B-Instruct-2506 (Multimodal/VLM)
Variante: Chain-of-Thought (CoT)
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

from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from transformers import Mistral3ForConditionalGeneration

# ============================================================================
# KONFIGURATION - MISTRAL SMALL 3.2 24B (CoT)
# ============================================================================

MODEL_NAME = "Mistral-Small-3.2-24B-CoT"
MODEL_HF_ID = "mistralai/Mistral-Small-3.2-24B-Instruct-2506"

SEED = 42
LOG_FILE = OUTPUT_DIR / f"{MODEL_NAME}_results.jsonl"
EXCEL_FILE = OUTPUT_DIR / f"{MODEL_NAME}_summary.xlsx"

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
        r"(?:antwort|answer|lösung|solution)[:\s]+([A-E])\b",
        r"\b([A-E])\s*(?:ist|is)\s+(?:richtig|correct)",
        r'"answer"\s*:\s*"?([A-E])"?'
    ]
    for p in patterns:
        m = re.search(p, clean_text, re.IGNORECASE)
        if m:
            return {"prediction": m.group(1).upper(), "format_valid": False, "error": "Regex Extraction"}

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
# EVALUATOR (MISTRAL SMALL 3.2 - CoT)
# ============================================================================

class VLMEvaluator:
    def __init__(self):
        logger.info(f"Lade {MODEL_NAME} ({MODEL_HF_ID})")

        # Tokenizer laden
        self.tokenizer = MistralTokenizer.from_hf_hub(MODEL_HF_ID)

        # Modell laden (BF16)
        self.model = Mistral3ForConditionalGeneration.from_pretrained(
            MODEL_HF_ID,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()
        
        logger.info(f"{MODEL_NAME} bereit")

    def generate(self, image_path: str) -> Dict:
        full_path = DATA_DIR / image_path
        if not full_path.exists():
            logger.warning(f"Bild nicht gefunden: {full_path}")
            return {"error": "Image not found", "prediction": None}

        # Bild laden und zu Data URL konvertieren
        image = Image.open(full_path).convert("RGB")
        
        # Bild als base64 Data URL
        import base64
        from io import BytesIO
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        image_url = f"data:image/png;base64,{img_base64}"

        # CHAIN-OF-THOUGHT PROMPT (wie bei anderen CoT-Modellen)
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

        # Mistral Message Format
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        ]

        # Tokenisieren mit MistralTokenizer
        tokenized = self.tokenizer.encode_chat_completion(ChatCompletionRequest(messages=messages))
        
        input_ids = torch.tensor([tokenized.tokens]).to(self.model.device)
        attention_mask = torch.ones_like(input_ids)
        
        # Bilder verarbeiten
        if tokenized.images:
            pixel_values = torch.tensor(tokenized.images[0], dtype=torch.bfloat16).unsqueeze(0).to(self.model.device)
            image_sizes = torch.tensor([pixel_values.shape[-2:]]).to(self.model.device)
        else:
            pixel_values = None
            image_sizes = None

        start_time = time.time()
        with torch.inference_mode():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_sizes=image_sizes,
                max_new_tokens=128,
                do_sample=False,
            )
        duration = time.time() - start_time

        # Nur neue Tokens dekodieren
        output_text = self.tokenizer.decode(outputs[0][len(tokenized.tokens):])

        result = parse_response(output_text)

        return {
            "prediction": result["prediction"],
            "format_valid": result["format_valid"],
            "error": result["error"],
            "inference_time": round(duration, 4),
            "input_tokens": len(tokenized.tokens),
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
