#!/usr/bin/env python3
"""
VLM Benchmark für Känguru-Mathematik-Aufgaben
Modell: moonshotai/Kimi-VL-A3B-Thinking-2506
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
# SETUP
# ----------------------------------------------------------------------------
_script_path = Path(__file__).resolve()
PROJECT_ROOT = Path(os.environ.get("VLM_PROJECT_ROOT", _script_path.parent.parent.parent.parent))

# Dotenv
try:
    from dotenv import load_dotenv
    _env_file = PROJECT_ROOT / ".env"
    if _env_file.exists(): load_dotenv(_env_file)
    else: load_dotenv()
except ImportError: pass

# HF Login
from huggingface_hub import login
HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN: login(token=HF_TOKEN)

# vLLM
from vllm import LLM, SamplingParams
try:
    from vllm.sampling_params import StructuredOutputsParams
    VLLM_HAS_STRUCTURED_OUTPUTS = True
except ImportError:
    VLLM_HAS_STRUCTURED_OUTPUTS = False

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
MODEL_NAME = "Kimi-VL-A3B"
MODEL_HF_ID = "moonshotai/Kimi-VL-A3B-Thinking-2506" 

DATASET_PATH = PROJECT_ROOT / "dataset_final.json"
# Fallback Dataset Suche
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
    answer: AnswerChoice = Field(description="Die Antwort: A-E")

ANSWER_JSON_SCHEMA = KanguruAnswer.model_json_schema()

# ----------------------------------------------------------------------------
# HELPERS
# ----------------------------------------------------------------------------
def set_seed(seed: int): random.seed(seed)

def parse_response(output_text: str) -> Dict:
    clean = output_text.strip()
    try:
        return {"prediction": json.loads(clean)["answer"], "format_valid": True, "error": None}
    except: pass
    m = re.search(r"\{[^{}]*\}", clean, re.DOTALL)
    if m:
        try:
            return {"prediction": json.loads(m.group(0))["answer"], "format_valid": True, "error": None}
        except: pass
    for L in ["A", "B", "C", "D", "E"]:
        if L in clean.upper(): return {"prediction": L, "format_valid": False, "error": "Extracted"}
    return {"prediction": None, "format_valid": False, "error": "No valid answer"}

def load_image_base64(path: Path) -> str:
    with open(path, "rb") as f: return base64.b64encode(f.read()).decode("utf-8")

# ----------------------------------------------------------------------------
# EVALUATOR
# ----------------------------------------------------------------------------
class VLMEvaluator:
    def __init__(self):
        logger.info(f"🏗️ Lade {MODEL_NAME} ({MODEL_HF_ID})")
        
        self.llm = LLM(
            model=MODEL_HF_ID,
            trust_remote_code=True,
            max_model_len=32768,  # Reduziert von 131072 auf 32K für weniger VRAM
            max_num_seqs=1,       # Reduziert von 8 auf 1 für weniger Parallelität
            limit_mm_per_prompt={"image": 256},  # Reduziert von 256 auf 128
            gpu_memory_utilization=0.85,  # Etwas weniger als 0.9
            dtype="bfloat16",
        )
        
        if VLLM_HAS_STRUCTURED_OUTPUTS:
            self.sampling_params = SamplingParams(
                max_tokens=1024, 
                temperature=0.0,
                structured_outputs=StructuredOutputsParams(json=ANSWER_JSON_SCHEMA)
            )
        else:
            self.sampling_params = SamplingParams(max_tokens=1024, temperature=0.0)

    def generate(self, image_path: str) -> Dict:
        p = DATA_DIR / image_path
        if not p.exists(): return {"error": "Image missing", "prediction": None}
        
        b64 = load_image_base64(p)
        messages = [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                {"type": "text", "text": "Löse die Aufgabe. Gib nur das JSON {'answer': 'X'} zurück."}
            ]
        }]
        
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
    if not DATASET_PATH.exists(): return
    data = json.load(open(DATASET_PATH))
    processed = set()
    if LOG_FILE.exists():
        processed = {json.loads(line).get("task_id") for line in open(LOG_FILE) if line.strip()}
    
    queue = [d for d in data if f"{d.get('year')}_{d.get('class')}_{d.get('task_id')}" not in processed]
    if not queue: return

    evaluator = VLMEvaluator()
    try:
        with open(LOG_FILE, 'a') as f:
            pbar = tqdm(queue, desc=MODEL_NAME)
            correct = 0
            for i, item in enumerate(pbar):
                tid = f"{item.get('year')}_{item.get('class')}_{item.get('task_id')}"
                try:
                    res = evaluator.generate(item["image_path"])
                    ok = res["prediction"] == item["answer"]
                    if ok: correct += 1
                    
                    log = {
                        "task_id": tid,
                        "ground_truth": item["answer"],
                        "prediction": res["prediction"],
                        "is_correct": ok,
                        "raw": res.get("raw_output"),
                        **item
                    }
                    f.write(json.dumps(log) + "\n"); f.flush()
                    pbar.set_postfix({"acc": f"{correct/(i+1):.1%}"})
                except Exception as e:
                    logger.error(f"{tid}: {e}")
                    if "out of memory" in str(e).lower(): break
    finally:
        evaluator.cleanup()

if __name__ == "__main__":
    run_benchmark()
