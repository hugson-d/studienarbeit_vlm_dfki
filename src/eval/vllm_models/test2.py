#!/usr/bin/env python3
"""
VLM Benchmark für Qwen3-VL (State-of-the-Art Dez 2025)
Modell: Qwen/Qwen3-VL-8B-Instruct
"""

import os
import json
import logging
import re
import time
import random
import gc
import base64
import sys
from pathlib import Path
from enum import Enum
from typing import Dict, List, Union

import pandas as pd
from tqdm import tqdm
from pydantic import BaseModel, Field

# ----------------------------------------------------------------------------
# HACK: OpenCV Patch
# ----------------------------------------------------------------------------
try:
    import cv2
except ImportError:
    import sys
    try:
        from cv2 import cv2
        sys.modules['cv2'] = cv2
    except ImportError:
        pass

# ----------------------------------------------------------------------------
# SETUP
# ----------------------------------------------------------------------------
_script_path = Path(__file__).resolve()
PROJECT_ROOT = Path(os.environ.get("VLM_PROJECT_ROOT", _script_path.parent.parent.parent.parent))

try:
    from dotenv import load_dotenv
    _env_file = PROJECT_ROOT / ".env"
    if _env_file.exists(): load_dotenv(_env_file)
    else: load_dotenv()
except ImportError: pass

from huggingface_hub import login
HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN: login(token=HF_TOKEN)

from vllm import LLM, SamplingParams
try:
    from vllm.sampling_params import StructuredOutputsParams
    VLLM_HAS_STRUCTURED_OUTPUTS = True
except ImportError:
    VLLM_HAS_STRUCTURED_OUTPUTS = False

# ----------------------------------------------------------------------------
# CONFIG: QWEN 3
# ----------------------------------------------------------------------------
MODEL_NAME = "Qwen3-VL-8B"
MODEL_HF_ID = "Qwen/Qwen3-VL-8B-Instruct"

# Cache Verzeichnis
MODEL_CACHE_DIR = os.environ.get("HF_HOME", "/netscratch/$USER/.cache/huggingface")

DATASET_PATH = PROJECT_ROOT / "dataset_final.json"
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
logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(), logging.FileHandler(OUTPUT_DIR / f"{MODEL_NAME}.log")])
logger = logging.getLogger(MODEL_NAME)

# ----------------------------------------------------------------------------
# SCHEMA
# ----------------------------------------------------------------------------
class AnswerChoice(str, Enum):
    A = "A"; B = "B"; C = "C"; D = "D"; E = "E"

class KanguruAnswer(BaseModel):
    answer: AnswerChoice = Field(description="Die korrekte Antwort (A, B, C, D oder E).")

ANSWER_JSON_SCHEMA = KanguruAnswer.model_json_schema()

# ----------------------------------------------------------------------------
# HELPER
# ----------------------------------------------------------------------------
def set_seed(seed: int): random.seed(seed)

def parse_response(output_text: str) -> Dict:
    # Qwen3 hat oft <think> Tags. Wir entfernen sie für das JSON Parsing,
    # behalten sie aber für das Logging eventuell interessant.
    clean = re.sub(r'<think>.*?</think>', '', output_text, flags=re.DOTALL).strip()
    clean = clean.replace("```json", "").replace("```", "").strip()

    try:
        return {"prediction": json.loads(clean)["answer"], "format_valid": True, "error": None}
    except: pass
    
    m = re.search(r"\{[^{}]*\}", clean, re.DOTALL)
    if m:
        try:
            return {"prediction": json.loads(m.group(0))["answer"], "format_valid": True, "error": None}
        except: pass
        
    return {"prediction": None, "format_valid": False, "error": "No valid JSON"}

def load_image_base64(path: Path) -> str:
    with open(path, "rb") as f: return base64.b64encode(f.read()).decode("utf-8")

def get_image_mime_type(path: Path) -> str:
    return "image/png" if path.suffix.lower() == ".png" else "image/jpeg"

# ----------------------------------------------------------------------------
# EVALUATOR
# ----------------------------------------------------------------------------
class VLMEvaluator:
    def __init__(self):
        logger.info(f"🏗️ Lade {MODEL_NAME} ({MODEL_HF_ID})")
        
        # Qwen3-VL Konfiguration
        self.llm = LLM(
            model=MODEL_HF_ID,
            trust_remote_code=True,
            max_model_len=32768,        # Qwen3 kann bis 1M, aber 32k spart VRAM
            gpu_memory_utilization=0.92,
            limit_mm_per_prompt={"image": 1},
            dtype="bfloat16",
        )
        
        # Sampling Params
        if VLLM_HAS_STRUCTURED_OUTPUTS:
            # Wir nutzen Structured Outputs, um das "Thinking" in valides JSON zu zwingen
            self.sampling_params = SamplingParams(
                max_tokens=1024, # Genug Platz falls er doch "denkt"
                temperature=0.0,
                structured_outputs=StructuredOutputsParams(json=ANSWER_JSON_SCHEMA)
            )
        else:
            self.sampling_params = SamplingParams(max_tokens=1024, temperature=0.0)

    def generate(self, image_path: str) -> Dict:
        p = DATA_DIR / image_path
        if not p.exists(): return {"error": "Image missing", "prediction": None}
        
        b64 = load_image_base64(p)
        mime = get_image_mime_type(p)
        
        # Prompt Engineering für Qwen3
        # Wir weisen es explizit an, NICHT zu "thinken" wenn wir nur JSON wollen,
        # oder wir lassen es zu, parsen aber robuster.
        system_prompt = (
            "Du bist ein Experte für Mathematik-Wettbewerbe. "
            "Analysiere die Aufgabe im Bild schrittweise, aber gib am Ende NUR das JSON aus."
        )
        
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                    {"type": "text", "text": "Löse die Aufgabe. Ausgabeformat: {\"answer\": \"X\"} mit X aus [A,B,C,D,E]."}
                ]
            }
        ]
        
        t0 = time.time()
        out = self.llm.chat(messages=messages, sampling_params=self.sampling_params, use_tqdm=False)
        dt = time.time() - t0
        
        txt = out[0].outputs[0].text
        in_tok = len(out[0].prompt_token_ids) if out[0].prompt_token_ids else 0
        
        r = parse_response(txt)
        return {**r, "inference_time": round(dt, 4), "input_tokens": in_tok, "raw_output": txt}

    def cleanup(self):
        if hasattr(self, 'llm'): del self.llm
        gc.collect()

# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------
def run_benchmark():
    set_seed(SEED)
    data = json.load(open(DATASET_PATH)) if DATASET_PATH.exists() else []
    if not data: return
    
    processed = set()
    if LOG_FILE.exists():
        processed = {json.loads(line).get("task_id") for line in open(LOG_FILE) if line.strip()}
    
    evaluator = VLMEvaluator()
    try:
        with open(LOG_FILE, 'a') as f:
            pbar = tqdm(data, desc=MODEL_NAME)
            correct = 0; count = 0
            for item in pbar:
                tid = f"{item.get('year')}_{item.get('class')}_{item.get('task_id')}"
                if tid in processed: continue
                
                try:
                    res = evaluator.generate(item["image_path"])
                    ok = res["prediction"] == item["answer"]
                    if ok: correct += 1
                    count += 1
                    
                    log = {
                        "model": MODEL_NAME, "task_id": tid, "ground_truth": item["answer"],
                        "prediction": res["prediction"], "is_correct": ok, **res,
                        "year": item.get("year"), "class": item.get("class"), "math_category": item.get("math_category")
                    }
                    f.write(json.dumps(log) + "\n"); f.flush()
                    pbar.set_postfix({"acc": f"{correct/count:.1%}"})
                except Exception as e:
                    logger.error(f"{tid}: {e}")
                    if "out of memory" in str(e).lower(): break
    finally:
        evaluator.cleanup()

if __name__ == "__main__":
    run_benchmark()