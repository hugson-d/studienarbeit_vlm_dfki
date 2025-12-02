#!/usr/bin/env python3
"""
VLM Benchmark für Känguru-Mathematik-Aufgaben
Modell: HuggingFaceM4/Idefics3-8B-Llama3 (vLLM Backend)
"""

import os
import json
import logging
import re
import time
import random
import gc
import pandas as pd
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

from vllm import LLM, SamplingParams
from vllm.assets.image import ImageAsset

# ============================================================================
# KONFIGURATION - IDEFICS3
# ============================================================================

MODEL_NAME = "Idefics3-8B-Llama3-vLLM"
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

# ============================================================================
# UTILS & PARSING
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

    # Regex Fallback
    patterns = [
        r'(?:antwort|answer|lösung|solution)[:\s]+([A-E])\b',
        r'\b([A-E])\s*(?:ist|is)\s+(?:richtig|correct)',
        r'"answer"\s*:\s*"?([A-E])"?'
    ]
    for p in patterns:
        m = re.search(p, clean_text, re.IGNORECASE)
        if m: return {"prediction": m.group(1).upper(), "format_valid": False, "error": "Regex Extraction"}
    
    # Letzter Fallback: Suche nach dem letzten Vorkommen eines Buchstabens A-E
    last_letter_match = re.findall(r'\b([A-E])\b', clean_text.upper())
    if last_letter_match:
        return {"prediction": last_letter_match[-1], "format_valid": False, "error": "Fallback: Last A-E"}
    
    return {"prediction": None, "format_valid": False, "error": "No valid answer"}

def set_seed(seed):
    random.seed(seed)

def free_gpu_memory():
    gc.collect()

# ============================================================================
# EVALUATOR
# ============================================================================

class VLMEvaluator:
    def __init__(self):
        logger.info(f"🏗️ Lade {MODEL_NAME} mit vLLM...")
        
        # vLLM Initialisierung
        self.llm = LLM(
            model=MODEL_HF_ID,
            trust_remote_code=True,
            max_model_len=4096,
            limit_mm_per_prompt={"image": 1},
            gpu_memory_utilization=0.9,
        )
        
        # Sampling Parameters (Greedy Decoding für Vergleichbarkeit)
        self.sampling_params = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            max_tokens=128,
            stop_token_ids=None,
        )
        
        logger.info(f"✅ {MODEL_NAME} bereit")

    def generate(self, image_path: str) -> Dict:
        full_path = DATA_DIR / image_path
        if not full_path.exists():
            return {"error": "Image not found", "prediction": None}

        # PROMPT ENGINEERING
        system_prompt = (
            "Du bist ein präzises mathematisches Assistenzsystem.\n\n"
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

        # vLLM Format: Text mit <image> Platzhalter
        prompt = f"{system_prompt}\n\n<image>\n\n{user_prompt}"

        try:
            start_time = time.time()
            
            # vLLM Inference
            outputs = self.llm.generate(
                {
                    "prompt": prompt,
                    "multi_modal_data": {"image": str(full_path)},
                },
                sampling_params=self.sampling_params
            )
            
            duration = time.time() - start_time
            
            # Output extrahieren
            output_text = outputs[0].outputs[0].text
            
            # Parsing
            result = parse_response(output_text)
            
            return {
                "prediction": result["prediction"],
                "format_valid": result["format_valid"],
                "error": result["error"],
                "inference_time": round(duration, 4)
            }
            
        except Exception as e:
            logger.error(f"Fehler bei Inference: {e}")
            return {"error": str(e), "prediction": None, "inference_time": 0}

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
