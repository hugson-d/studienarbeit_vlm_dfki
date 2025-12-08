#!/usr/bin/env python3
"""
VLM Benchmark für Känguru-Mathematik-Aufgaben
Modell: HuggingFaceM4/Idefics3-8B-Llama3 mit vLLM-Backend
- Chat-Template via AutoProcessor.apply_chat_template
- Structured Outputs (JSON Schema) via StructuredOutputsParams
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

# HuggingFace Login
from huggingface_hub import login
HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN:
    login(token=HF_TOKEN)

# vLLM + Structured Outputs + HF Processor
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams
from transformers import AutoProcessor

# ============================================================================
# KONFIGURATION - IDEFICS3 + vLLM
# ============================================================================

MODEL_NAME = "Idefics3-8B-Llama3-vLLM-structured"
MODEL_HF_ID = "HuggingFaceM4/Idefics3-8B-Llama3"
MODEL_PARAMS_B = 8

# Projekt-Root
_script_path = Path(__file__).resolve()
PROJECT_ROOT = Path(os.environ.get("VLM_PROJECT_ROOT", _script_path.parent.parent.parent))

DATASET_PATH = PROJECT_ROOT / "dataset_final.json"
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "evaluation_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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
# SCHEMA / STRUCTURED OUTPUTS
# ============================================================================

class KanguruAnswer(BaseModel):
    answer: Literal["A", "B", "C", "D", "E"] = Field(
        description="Die korrekte Antwort: A, B, C, D oder E."
    )

ANSWER_JSON_SCHEMA = KanguruAnswer.model_json_schema()

# ============================================================================
# UTILS & PARSING
# ============================================================================

def parse_response(output_text: str) -> Dict:
    """
    Parsing für structured outputs:
    - Primär: direktes JSON-Parsing + Pydantic-Validierung (sollte durch vLLM garantiert sein)
    - Sekundär: JSON-Objekt im Text suchen
    - Letzter Fallback: letztes standalone-[A-E] (nur für Evaluationszwecke, format_valid=False)
    """
    clean_text = output_text.strip()

    # Markdown-Codeblock entfernen, falls das Modell doch sowas macht
    if "```" in clean_text:
        match = re.search(r'```(?:json)?\s*(.*?)\s*```', clean_text, re.DOTALL)
        if match:
            clean_text = match.group(1).strip()

    # 1) Direkter JSON-Parse (Sollfall bei Structured Outputs)
    try:
        data = json.loads(clean_text)
        # Keys normalisieren (answer vs ANSWER)
        if isinstance(data, dict):
            data = {k.lower(): v for k, v in data.items()}
        validated = KanguruAnswer(**data)
        return {
            "prediction": validated.answer,
            "format_valid": True,
            "error": None,
        }
    except Exception:
        pass

    # 2) JSON-Objekt aus Text extrahieren
    json_match = re.search(r'\{[^{}]*\}', clean_text, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group(0))
            if isinstance(data, dict):
                data = {k.lower(): v for k, v in data.items()}
            validated = KanguruAnswer(**data)
            return {
                "prediction": validated.answer,
                "format_valid": False,
                "error": "JSON extracted from surrounding text",
            }
        except Exception:
            pass

    # 3) Letzter Fallback: letztes standalone-[A-E] (für Evaluation, nicht „echtes“ structured output)
    last_letter_match = re.findall(r'\b([A-E])\b', clean_text.upper())
    if last_letter_match:
        return {
            "prediction": last_letter_match[-1],
            "format_valid": False,
            "error": "Fallback: last standalone A-E",
        }

    # Gar nichts brauchbares gefunden
    return {
        "prediction": None,
        "format_valid": False,
        "error": "No valid answer / JSON",
    }

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
# EVALUATOR (vLLM + HF-Chat-Template + Structured Outputs)
# ============================================================================

class VLMEvaluator:
    def __init__(self):
        logger.info(f"🏗️ Lade {MODEL_NAME} ({MODEL_PARAMS_B}B) mit vLLM")
        logger.info(f"   HuggingFace ID: {MODEL_HF_ID}")
        logger.info(f"   ✅ Structured Outputs (JSON Schema) aktiviert")

        # 1. HF Processor: nur für Chat-Template
        self.processor = AutoProcessor.from_pretrained(MODEL_HF_ID)

        # 2. vLLM LLM (Multimodal)
        #    mm_processor_kwargs an Idefics3 weiterreichen (laut vLLM-Doku)
        self.llm = LLM(
            model=MODEL_HF_ID,
            trust_remote_code=True,
            max_model_len=8192,
            gpu_memory_utilization=0.9,
            dtype="bfloat16",
            limit_mm_per_prompt={"image": 1},
            mm_processor_kwargs={
                "size": {
                    # siehe HF-Model-Card für Idefics3 („longest_edge“)
                    "longest_edge": 3 * 364
                },
            },
        )

        # 3. Structured Outputs konfigurieren
        structured_outputs = StructuredOutputsParams(
            json=ANSWER_JSON_SCHEMA
        )

        # 4. Sampling Params
        self.sampling_params = SamplingParams(
            max_tokens=512,
            temperature=0.0,
            structured_outputs=structured_outputs,
        )

        logger.info(f"✅ vLLM + StructuredOutputsParams initialisiert")
        logger.info(f"   JSON Schema: {json.dumps(ANSWER_JSON_SCHEMA, ensure_ascii=False)}")

    def generate(self, image_path: str) -> Dict:
        full_path = DATA_DIR / image_path
        if not full_path.exists():
            logger.error(f"❌ Bild nicht gefunden: {full_path}")
            logger.error(f"   DATA_DIR: {DATA_DIR}")
            logger.error(f"   image_path: {image_path}")
            return {
                "error": f"Image not found: {full_path}",
                "prediction": None,
                "format_valid": False,
                "inference_time": 0.0,
                "input_tokens": 0,
            }

        # Bild laden (PIL) – vLLM übernimmt Vorverarbeitung
        image = Image.open(full_path).convert("RGB")

        # Prompt wie in deinem HF-Skript – Systeminstruktion in Text eingebettet
        system_prompt = (
            "Du bist ein mathematisches Assistenzsystem für Multiple-Choice-Aufgaben.\n"
            "Analysiere das Bild und wähle die korrekte Antwort: A, B, C, D oder E.\n\n"
            "Antworte im JSON-Format: {\"answer\": \"X\"} wobei X = A, B, C, D oder E."
        )
        user_prompt = "Bestimme die korrekte Antwort basierend auf dem Bild. Gib nur das JSON zurück."

        # Messages im HF-Format – identisch zur funktionierenden HF-Version
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": f"{system_prompt}\n\n{user_prompt}"}
                ]
            }
        ]

        # Chat-Template anwenden → reiner String-Prompt für vLLM
        prompt = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False
        )

        # vLLM-Generierung: Prompt + Bild als multi_modal_data
        start_time = time.time()
        outputs = self.llm.generate(
            {
                "prompt": prompt,
                "multi_modal_data": {
                    "image": image
                },
            },
            sampling_params=self.sampling_params,
            use_tqdm=False
        )
        duration = time.time() - start_time

        generated_text = outputs[0].outputs[0].text
        input_tokens = len(outputs[0].prompt_token_ids) if outputs[0].prompt_token_ids else 0

        result = parse_response(generated_text)

        return {
            "prediction": result["prediction"],
            "format_valid": result["format_valid"],
            "error": result["error"],
            "inference_time": round(duration, 4),
            "input_tokens": input_tokens,
        }

    def cleanup(self):
        del self.llm
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

    with open(DATASET_PATH, encoding="utf-8") as f:
        dataset = json.load(f)

    # Resume-Logik
    processed_ids = set()
    if LOG_FILE.exists():
        with open(LOG_FILE, "r", encoding="utf-8") as f:
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
    if "format_valid" in df.columns:
        print(f"  Valid JSON:   {df['format_valid'].mean():.1%}")

    if "math_category" in df.columns:
        print("\n📐 Nach Kategorie:")
        print(df.groupby("math_category")["is_correct"].mean().apply(lambda x: f"{x:.1%}"))

if __name__ == "__main__":
    run_benchmark()
