#!/usr/bin/env python3
"""
VLM Benchmark für Känguru-Mathematik-Aufgaben
Modell: Google Gemma-3-27b-it (Multimodal/VLM)
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
    AutoModelForMultimodalLM, # Korrekte Klasse für Gemma-3
    BitsAndBytesConfig
)

# ============================================================================
# KONFIGURATION - GEMMA 3
# ============================================================================

MODEL_NAME = "Gemma-3-4b-it"
MODEL_HF_ID = "google/gemma-3-4b-it" # Sicherstellen, dass die ID korrekt ist
MODEL_PARAMS_B = 4

# Quantisierung: 27B ist groß. Auf Consumer-Hardware (3090/4090) zwingend 4-Bit.
# Auf A100 (80GB) könnte bfloat16 passen. Wir nutzen hier die Logik:
USE_QUANTIZATION = False # Erzwingen für 27B, falls VRAM < 60GB

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

# ============================================================================
# UTILS & PARSING (Identisch zu Qwen für Vergleichbarkeit)
# ============================================================================

class KanguruAnswer(BaseModel):
    answer: Literal['A', 'B', 'C', 'D', 'E'] = Field(description="Die korrekte Antwort.")

def parse_response(output_text: str) -> Dict:
    clean_text = output_text.strip()
    # Markdown Code-Block entfernen
    if "```" in clean_text:
        match = re.search(r'```(?:json)?\s*(.*?)\s*```', clean_text, re.DOTALL)
        if match: clean_text = match.group(1).strip()
    
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

    # Regex Fallback (Gleiche Patterns wie Qwen)
    patterns = [
        r'(?:antwort|answer|lösung|solution)[:\s]+([A-E])\b',
        r'\b([A-E])\s*(?:ist|is)\s+(?:richtig|correct)',
        r'"answer"\s*:\s*"?([A-E])"?'
    ]
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
# EVALUATOR
# ============================================================================

class VLMEvaluator:
    def __init__(self):
        logger.info(f"🏗️ Lade {MODEL_NAME} ({MODEL_PARAMS_B}B)")
        logger.info(f"   Quantisierung (4-bit): {USE_QUANTIZATION}")

        bnb_config = None
        if USE_QUANTIZATION:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True, 
                bnb_4bit_quant_type="nf4", 
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True
            )

        # 1. Processor
        # Gemma VLM nutzt typischerweise PaliGemmaProcessor oder AutoProcessor
        try:
            self.processor = AutoProcessor.from_pretrained(MODEL_HF_ID, trust_remote_code=True)
        except Exception as e:
            logger.error(f"Fehler beim Laden des Processors: {e}")
            raise e

        # 2. Modell
        load_kwargs = {
            "device_map": "auto",
            "trust_remote_code": True,
            "torch_dtype": torch.bfloat16,
            "low_cpu_mem_usage": True
        }
        if bnb_config:
            load_kwargs["quantization_config"] = bnb_config

        # Flash Attention Check
        try:
            import flash_attn
            load_kwargs["attn_implementation"] = "flash_attention_2"
            logger.info("   ⚡ Flash Attention 2 aktiviert")
        except ImportError:
            pass

        # AutoModel lädt automatisch die korrekte Architektur (z.B. PaliGemmaForConditionalGeneration)
        self.model = AutoModelForMultimodalLM.from_pretrained(MODEL_HF_ID, **load_kwargs).eval()
        
        logger.info(f"✅ {MODEL_NAME} bereit auf {self.model.device}")

    def generate(self, image_path: str) -> Dict:
        full_path = DATA_DIR / image_path
        if not full_path.exists():
            return {"error": "Image not found", "prediction": None}

        # Bild laden
        image = Image.open(full_path).convert("RGB")
        
        # PROMPT ENGINEERING (Exakt wie Qwen/InternVL)
        system_prompt = (
            "Du bist ein präzises mathematisches Assistenzsystem.\n\n"
            "Deine Aufgabe ist es, ausschließlich das Endergebnis zu liefern.\n"
            "Deine Ausgabe MUSS aus einem einzigen JSON-Objekt bestehen.\n\n"
            "GÜLTIGES FORMAT:\n"
            '{"answer": "X"}\n'
            "wobei X einer der Buchstaben A, B, C, D oder E ist.\n\n"
            "STRIKTE REGELN:\n"
            "- KEINE Erklärungen, Herleitungen oder Rechenwege.\n"
            "- KEIN Text vor oder nach dem JSON.\n"
            "- KEIN Markdown (außer für den JSON-Block falls nötig).\n"
            "- Wenn die Lösung unklar ist, wähle die wahrscheinlichste Option (A-E).\n"
            "- Das JSON muss syntaktisch valide sein."
        )
        user_prompt = "Bestimme die korrekte Antwort basierend auf dem Bild. Gib nur das JSON zurück."

        # Chat Template Erstellung
        # Dies ist der moderne Weg für Gemma/Llama VLMs
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": f"{system_prompt}\n\n{user_prompt}"}
                ]
            }
        ]

        # Vorbereitung der Inputs
        inputs = self.processor.apply_chat_template(
            messages, 
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(self.model.device)

        # Generierung
        start_time = time.time()
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=128, # Genug Platz für JSON
                do_sample=False,    # Greedy decoding (WICHTIG für Vergleichbarkeit)
                temperature=0.0,
                # eos_token_id sicherstellen
                eos_token_id=self.processor.tokenizer.eos_token_id
            )
        duration = time.time() - start_time

        # Decoding
        input_len = inputs.input_ids.shape[1]
        trimmed_ids = generated_ids[:, input_len:]
        output_text = self.processor.batch_decode(
            trimmed_ids, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )[0]

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
                try: processed_ids.add(json.loads(line)['task_id']) 
                except: pass

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
                
                if is_correct: correct_count += 1
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
    if not LOG_FILE.exists(): return
    df = pd.read_json(LOG_FILE, lines=True)
    if df.empty: return
    
    df.to_excel(EXCEL_FILE, index=False)
    
    print("\n" + "="*70)
    print(f"📊 ERGEBNISSE: {MODEL_NAME}")
    print(f"  Accuracy:     {df['is_correct'].mean():.1%}")
    print(f"  Valid JSON:   {df['format_valid'].mean():.1%}")
    
    if 'math_category' in df.columns:
        print("\n📐 Nach Kategorie:")
        print(df.groupby('math_category')['is_correct'].mean().apply(lambda x: f"{x:.1%}"))

if __name__ == "__main__":
    run_benchmark()