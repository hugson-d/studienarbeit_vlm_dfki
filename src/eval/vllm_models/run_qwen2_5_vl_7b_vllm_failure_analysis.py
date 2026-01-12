#!/usr/bin/env python3
import os

# Flash Attention Backend global deaktivieren, bevor vLLM geladen wird
# Gültige Optionen: FLASH_ATTN, TORCH_SDPA, TRITON_ATTN, etc.
os.environ["VLLM_ATTENTION_BACKEND"] = "TORCH_SDPA"

import json
import logging
import time
import random
import gc
import base64
import traceback
import pandas as pd
from collections import Counter, defaultdict
from typing import Dict, List, Union, Optional, Any
from pathlib import Path
from tqdm import tqdm
from enum import Enum
from pydantic import BaseModel, Field
from datetime import datetime

from vllm import LLM, SamplingParams
try:
    from vllm.sampling_params import StructuredOutputsParams
except ImportError:
    raise ImportError("vLLM Version zu alt. Bitte vLLM >= 0.6.0 installieren.")

# ============================================================================
# KONFIGURATION
# ============================================================================

N_RUNS = 5             
N_VOTING_PATHS = 1     
TEMPERATURE = 0.0      

BASE_MODEL_NAME = "Qwen2.5-VL-7B-Instruct"
MODEL_NAME = f"{BASE_MODEL_NAME}_FailureAnalysis_{N_RUNS}runs"
MODEL_HF_ID = "Qwen/Qwen2.5-VL-7B-Instruct"

_script_path = Path(__file__).resolve()
PROJECT_ROOT = Path(os.environ.get("VLM_PROJECT_ROOT", _script_path.parent.parent.parent.parent))
DATASET_PATH = PROJECT_ROOT / "dataset_final.json"
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "evaluation_results"
SEED = 42

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = OUTPUT_DIR / f"{MODEL_NAME}_{TIMESTAMP}_results.jsonl"
ERROR_LOG_FILE = OUTPUT_DIR / f"{MODEL_NAME}_{TIMESTAMP}_errors.jsonl"
SUMMARY_FILE = OUTPUT_DIR / f"{MODEL_NAME}_{TIMESTAMP}_summary.json"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler(OUTPUT_DIR / f"{MODEL_NAME}_{TIMESTAMP}.log")]
)
logger = logging.getLogger(MODEL_NAME)

# ============================================================================
# SCHEMA & UTILS
# ============================================================================

class AnswerChoice(str, Enum):
    A = "A"; B = "B"; C = "C"; D = "D"; E = "E"

class CoTResponse(BaseModel):
    reasoning: str = Field(description="Schritt-für-Schritt Herleitung.")
    answer: AnswerChoice = Field(description="Finaler Lösungsbuchstabe.")

COT_JSON_SCHEMA = CoTResponse.model_json_schema()

def set_seed(seed: int):
    random.seed(seed)

def load_image_base64(image_path: Path) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def analyze_raw_output(raw_text: str) -> Dict[str, Any]:
    analysis = {"raw_text": raw_text, "raw_length": len(raw_text) if raw_text else 0}
    if not raw_text:
        analysis["is_empty"] = True
        return analysis
    
    try:
        parsed = json.loads(raw_text)
        analysis["json_parse_success"] = True
        validated = CoTResponse(**parsed)
        analysis["pydantic_success"] = True
        analysis["answer_value"] = validated.answer.value
        analysis["reasoning_length"] = len(validated.reasoning)
    except Exception as e:
        analysis["json_parse_success"] = False
        analysis["pydantic_success"] = False
        analysis["error"] = str(e)
    return analysis

class ErrorCategory:
    EMPTY_OUTPUT = "empty_output"; INVALID_JSON = "invalid_json"
    PYDANTIC_VALIDATION = "pydantic_validation"; FILE_NOT_FOUND = "file_not_found"
    UNKNOWN = "unknown"

# ============================================================================
# EVALUATOR
# ============================================================================

class VLMEvaluatorWithLogging:
    def __init__(self):
        logger.info(f"🏗️ Initialisiere {MODEL_HF_ID} mit TORCH_SDPA Backend")
        self.llm = LLM(
            model=MODEL_HF_ID,
            trust_remote_code=True,
            max_model_len=8192, # Reduziert für Stabilität ohne Flash Attn
            gpu_memory_utilization=0.90,
            dtype="bfloat16",
            limit_mm_per_prompt={"image": 1},
        )
        self.sampling_params = SamplingParams(
            temperature=TEMPERATURE,
            max_tokens=1024,
            structured_outputs=StructuredOutputsParams(json=COT_JSON_SCHEMA),
        )

    def generate_single_run(self, image_path: str, task_id: str, run_id: int) -> Dict:
        full_path = DATA_DIR / image_path
        res = {"task_id": task_id, "run_id": run_id, "prediction": None}
        
        if not full_path.exists():
            res["error_category"] = ErrorCategory.FILE_NOT_FOUND
            return res
        
        try:
            image_b64 = load_image_base64(full_path)
            messages = [
                {"role": "system", "content": "Du bist ein Mathe-Tutor. Nutze Reasoning."},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
                    {"type": "text", "text": "Löse die Aufgabe aus dem Bild."}
                ]}
            ]
            
            outputs = self.llm.chat(messages=messages, sampling_params=self.sampling_params, use_tqdm=False)
            raw_text = outputs[0].outputs[0].text
            analysis = analyze_raw_output(raw_text)
            
            res.update({
                "raw_output": raw_text,
                "finish_reason": outputs[0].outputs[0].finish_reason,
                "raw_output_analysis": analysis
            })

            if analysis.get("pydantic_success"):
                res["prediction"] = analysis["answer_value"]
            else:
                res["error_category"] = ErrorCategory.INVALID_JSON
        except Exception as e:
            res["error_category"] = ErrorCategory.UNKNOWN
            res["error_details"] = str(e)
            
        return res

    def run_multiple_passes(self, image_path: str, task_id: str) -> List[Dict]:
        return [self.generate_single_run(image_path, task_id, i) for i in range(1, N_RUNS + 1)]

    def cleanup(self):
        if hasattr(self, 'llm'): del self.llm
        gc.collect()

def run_failure_analysis():
    set_seed(SEED)
    with open(DATASET_PATH, 'r') as f:
        dataset = json.load(f)

    evaluator = VLMEvaluatorWithLogging()
    all_results = []
    
    with open(LOG_FILE, 'w') as f_log:
        for item in tqdm(dataset):
            task_id = f"{item.get('year')}_{item.get('class')}_{item.get('task_id')}"
            run_results = evaluator.run_multiple_passes(item["image_path"], task_id)
            for r in run_results:
                f_log.write(json.dumps(r) + "\n")
                all_results.append(r)
    
    evaluator.cleanup()
    logger.info(f"Analyse beendet. Ergebnisse in {LOG_FILE}")

if __name__ == "__main__":
    run_failure_analysis()