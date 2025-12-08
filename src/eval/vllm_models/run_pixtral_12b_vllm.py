#!/usr/bin/env python3
"""
VLM Benchmark für Känguru-Mathematik-Aufgaben
Modell: Pixtral-12B-2409 (vLLM Backend mit Structured Outputs / JSON Schema)
"""

import os
import json
import logging
import re
import time
import random
import gc
import base64
from pathlib import Path
from enum import Enum
from typing import Dict, List, Union

import pandas as pd
from tqdm import tqdm
from pydantic import BaseModel, Field

# ----------------------------------------------------------------------------
# UMWELTVARIABLEN & SETUP
# ----------------------------------------------------------------------------

_script_path = Path(__file__).resolve()
# Falls VLM_PROJECT_ROOT nicht gesetzt, rate 4 Ebenen hoch
PROJECT_ROOT = Path(os.environ.get("VLM_PROJECT_ROOT", _script_path.parent.parent.parent.parent))

# Dotenv laden
try:
    from dotenv import load_dotenv
    _env_file = PROJECT_ROOT / ".env"
    if _env_file.exists():
        load_dotenv(_env_file)
    else:
        load_dotenv()
except ImportError:
    pass

# HF Login
from huggingface_hub import login
HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN:
    login(token=HF_TOKEN)

# vLLM Imports
from vllm import LLM, SamplingParams
try:
    from vllm.sampling_params import StructuredOutputsParams
    VLLM_HAS_STRUCTURED_OUTPUTS = True
except ImportError:
    VLLM_HAS_STRUCTURED_OUTPUTS = False

# ----------------------------------------------------------------------------
# KONFIGURATION
# ----------------------------------------------------------------------------

MODEL_NAME = "Pixtral-12B-vLLM"
MODEL_HF_ID = "mistralai/Pixtral-12B-2409" # KORREKTES VISION MODELL
MODEL_PARAMS_B = 12
MODEL_CACHE_DIR = os.environ.get("HF_HOME", "/netscratch/$USER/.cache/huggingface")

DATASET_PATH = PROJECT_ROOT / "dataset_final.json"
# Fallback Suche nach Dataset
if not DATASET_PATH.exists():
    _search = _script_path.parent
    for _ in range(5):
        if (_search / "dataset_final.json").exists():
            PROJECT_ROOT = _search
            DATASET_PATH = PROJECT_ROOT / "dataset_final.json"
            break
        _search = _search.parent

DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "evaluation_results"
LOG_FILE = OUTPUT_DIR / f"{MODEL_NAME}_results.jsonl"
SEED = 42

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler(OUTPUT_DIR / f"{MODEL_NAME}.log")]
)
logger = logging.getLogger(MODEL_NAME)

# ----------------------------------------------------------------------------
# SCHEMA DEFINITION
# ----------------------------------------------------------------------------

class AnswerChoice(str, Enum):
    A = "A"
    B = "B"
    C = "C"
    D = "D"
    E = "E"

class KanguruAnswer(BaseModel):
    answer: AnswerChoice = Field(description="Die Antwort: A-E")

ANSWER_JSON_SCHEMA = KanguruAnswer.model_json_schema()

# ----------------------------------------------------------------------------
# HILFSFUNKTIONEN
# ----------------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)

def parse_response(output_text: str) -> Dict[str, Union[str, bool, None]]:
    clean = output_text.strip()
    # 1. Versuch: Direktes JSON
    try:
        data = json.loads(clean)
        v = KanguruAnswer(**data)
        return {"prediction": v.answer.value, "format_valid": True, "error": None}
    except:
        pass
    
    # 2. Versuch: JSON im Text suchen
    m = re.search(r"\{[^{}]*\}", clean, re.DOTALL)
    if m:
        try:
            data = json.loads(m.group(0))
            v = KanguruAnswer(**data)
            return {"prediction": v.answer.value, "format_valid": True, "error": None}
        except:
            pass
            
    # 3. Versuch: Heuristik
    for L in ["A", "B", "C", "D", "E"]:
        if L in clean.upper():
            return {"prediction": L, "format_valid": False, "error": "Extracted"}
            
    return {"prediction": None, "format_valid": False, "error": "No valid answer"}

def load_image_base64(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def get_image_mime_type(path: Path) -> str:
    mt = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", 
          ".gif": "image/gif", ".webp": "image/webp", ".bmp": "image/bmp"}
    return mt.get(path.suffix.lower(), "image/jpeg")

# ----------------------------------------------------------------------------
# EVALUATOR KLASSE
# ----------------------------------------------------------------------------

class VLMEvaluator:
    def __init__(self):
        logger.info(f"🏗️ Lade {MODEL_NAME} ({MODEL_PARAMS_B}B)")
        
        # Pixtral Spezifische Konfiguration
        self.llm = LLM(
            model=MODEL_HF_ID,
            tokenizer_mode="mistral",         # Zwingend für Pixtral
            limit_mm_per_prompt={"image": 1}, # Zwingend für Pixtral
            trust_remote_code=True,
            max_model_len=32768,              # Reduziert von 128k für Stabilität
            gpu_memory_utilization=0.9,
            dtype="bfloat16",
        )
        
        if VLLM_HAS_STRUCTURED_OUTPUTS:
            logger.info("⚡ Structured Outputs aktiviert")
            structured_outputs = StructuredOutputsParams(json=ANSWER_JSON_SCHEMA)
            self.sampling_params = SamplingParams(
                max_tokens=512,
                temperature=0.0,
                structured_outputs=structured_outputs
            )
        else:
            logger.warning("⚠️ Structured Outputs nicht verfügbar - Fallback")
            self.sampling_params = SamplingParams(max_tokens=512, temperature=0.0)

    def generate(self, image_path: str) -> Dict:
        p = DATA_DIR / image_path
        if not p.exists():
            raise FileNotFoundError(str(p))
            
        b64 = load_image_base64(p)
        mime = get_image_mime_type(p)
        
        # Prompt Vorgabe (Beibehalten)
        system_prompt = ("Du bist ein mathematisches Assistenzsystem. Wähle A-E. "
                         "Antwortformat: {\"answer\":\"X\"}.")
        user_prompt = "Bestimme die richtige Antwort als JSON."
        
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user", 
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                    {"type": "text", "text": user_prompt}
                ]
            }
        ]
        
        t0 = time.time()
        # Generierung
        out = self.llm.chat(messages=messages, sampling_params=self.sampling_params, use_tqdm=False)
        dt = time.time() - t0
        
        txt = out[0].outputs[0].text
        input_tokens = len(out[0].prompt_token_ids) if out[0].prompt_token_ids else 0
        
        r = parse_response(txt)
        
        return {
            "prediction": r["prediction"],
            "format_valid": r["format_valid"],
            "error": r["error"],
            "inference_time": round(dt, 4),
            "input_tokens": input_tokens,
            "raw_output": txt
        }

    def cleanup(self):
        if hasattr(self, 'llm'):
            del self.llm
        gc.collect()

# ----------------------------------------------------------------------------
# BENCHMARK LOGIK
# ----------------------------------------------------------------------------

def load_dataset() -> List[Dict]:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset nicht gefunden: {DATASET_PATH}")
    with open(DATASET_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_processed_tasks() -> set:
    s = set()
    if LOG_FILE.exists():
        with open(LOG_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    e = json.loads(line)
                    s.add(e.get("task_id"))
                except:
                    pass
    return s

def create_task_id(item: Dict) -> str:
    return f"{item.get('year','unknown')}_{item.get('class','unknown')}_{item.get('task_id','unknown')}"

def run_benchmark():
    set_seed(SEED)
    data = load_dataset()
    processed = get_processed_tasks()
    
    # Check if finished
    remaining = [d for d in data if create_task_id(d) not in processed]
    if not remaining:
        logger.info("Alle Tasks bereits bearbeitet.")
        return

    evaluator = VLMEvaluator()
    correct = 0
    count = 0
    
    try:
        with open(LOG_FILE, 'a', encoding='utf-8') as f_log:
            pbar = tqdm(data, desc=MODEL_NAME, unit="task")
            for item in pbar:
                tid = create_task_id(item)
                if tid in processed:
                    continue
                
                try:
                    r = evaluator.generate(item["image_path"])
                    gt = item.get("answer")
                    
                    ok = (r["prediction"] == gt)
                    if ok: correct += 1
                    count += 1
                    
                    entry = {
                        "model": MODEL_NAME,
                        "task_id": tid,
                        "year": item.get("year"),
                        "class": item.get("class"),
                        "original_task_id": item.get("task_id"),
                        "math_category": item.get("math_category"),
                        "is_text_only": item.get("is_text_only"),
                        "ground_truth": gt,
                        "prediction": r["prediction"],
                        "is_correct": ok,
                        "format_valid": r["format_valid"],
                        "error_type": r["error"],
                        "inference_time": r["inference_time"],
                        "input_tokens": r["input_tokens"],
                        "raw_output": r.get("raw_output", "")
                    }
                    
                    f_log.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    f_log.flush()
                    
                    acc = correct / count if count > 0 else 0
                    last_status = '✓' if ok else '✗'
                    pbar.set_postfix({"acc": f"{acc:.1%}", "last": f"{last_status} GT:{gt} P:{r['prediction']}"})
                    
                except FileNotFoundError:
                    logger.warning(f"Bild fehlt: {tid}")
                except Exception as e:
                    logger.error(f"Error {tid}: {str(e)}")
                    if "out of memory" in str(e).lower():
                        logger.error("OOM erkannt - Abbruch")
                        break
            pbar.close()
    finally:
        evaluator.cleanup()

def generate_report():
    if not LOG_FILE.exists():
        return
    df = pd.read_json(LOG_FILE, lines=True)
    if df.empty:
        return
        
    print(f"\n{'='*40}\n📊 REPORT: {MODEL_NAME}\n{'='*40}")
    print(f"Total:      {len(df)}")
    print(f"Accuracy:   {df['is_correct'].mean():.1%}")
    print(f"Valid JSON: {df['format_valid'].mean():.1%}")
    print(f"Avg Time:   {df['inference_time'].mean():.2f}s")
    print(f"Log:        {LOG_FILE}")

if __name__ == "__main__":
    run_benchmark()
    generate_report()