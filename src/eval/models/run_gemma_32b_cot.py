#!/usr/bin/env python3
"""
VLM Benchmark für Känguru-Mathematik-Aufgaben
Modell: Google Gemma-3-27b-it (Multimodal/VLM) - CoT Version
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

MODEL_NAME = "Gemma-3-27b-it-CoT"
MODEL_HF_ID = "google/gemma-3-27b-it"
MODEL_PARAMS_B = 27

USE_QUANTIZATION = False

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

        self.model = AutoModelForMultimodalLM.from_pretrained(MODEL_HF_ID, **load_kwargs).eval()
        
        logger.info(f"✅ {MODEL_NAME} bereit auf {self.model.device}")

    def generate(self, image_path: str) -> Dict:
        full_path = DATA_DIR / image_path
        if not full_path.exists():
            return {"error": "Image not found", "prediction": None}

        # Bild laden
        image = Image.open(full_path).convert("RGB")
        
        # CONCISE CoT PROMPT mit strikter JSON-Ausgabe
        system_prompt = (
            "Du bist ein präzises mathematisches Assistenzsystem.\n\n"
            "ARBEITSWEISE:\n"
            "1. Denke Schritt für Schritt, aber nutze NUR Stichpunkte.\n"
            "2. Verwende Formeln statt Text wo möglich.\n"
            "3. Fasse dich extrem kurz.\n\n"
            "AUSGABEFORMAT:\n"
            "Deine Ausgabe MUSS ausschließlich aus einem einzigen JSON-Objekt bestehen.\n"
            "Keine Erklärungen. Keine Analyse. Kein Text davor oder danach.\n\n"
            "Das einzige gültige Ausgabeformat ist exakt:\n\n"
            '{"answer": "X"}\n\n'
            "wobei X genau einer der Buchstaben A, B, C, D oder E ist.\n\n"
            "Verboten:\n"
            "- Zusätzlicher Text\n"
            "- Kommentare\n"
            "- Markdown\n"
            "- Codeblöcke\n"
            "- Mehrere JSON-Objekte\n"
            "- Begründungen\n"
            "- Alternative Antworten\n"
            "- Leerzeilen vor oder nach dem JSON\n\n"
            "Wenn du keine Antwort findest, MUSST du trotzdem einen der Buchstaben A–E ausgeben.\n"
            "Das JSON MUSS syntaktisch korrekt sein."
        )
        user_prompt = "Löse die Mathematik-Aufgabe im Bild. Gib nur das JSON zurück."

        # Chat Template Erstellung
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
                max_new_tokens=512, # Mehr Tokens für CoT
                do_sample=False,    # Greedy decoding (WICHTIG für Vergleichbarkeit)
                temperature=0.0,
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
            "inference_time": round(duration, 4)
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
                    "math_category": task.get("math_category"),
                    "ground_truth": gt,
                    "prediction": result["prediction"],
                    "is_correct": is_correct,
                    "format_valid": result.get("format_valid"),
                    "raw_output": result.get("raw_output"),
                    "inference_time": result.get("inference_time")
                }
                
                f_log.write(json.dumps(log_entry) + "\n")
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
