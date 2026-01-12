#!/usr/bin/env python3
"""
VLM Benchmark: Känguru-Mathematik
Modell: InternVL3-14B (vLLM Backend)
Methode: Failure Analysis - 5 runs per task with detailed error logging
Status: Adapted from Qwen failure analysis
"""

import os
import json
import logging
import re
import time
import random
import gc
import traceback
import pandas as pd
from typing import Dict, List, Union, Optional, Any
from pathlib import Path
from tqdm import tqdm
from enum import Enum
from PIL import Image
from collections import Counter, defaultdict
from datetime import datetime

from pydantic import BaseModel, Field

# Projekt-Root
_script_path = Path(__file__).resolve()
PROJECT_ROOT = Path(os.environ.get("VLM_PROJECT_ROOT", _script_path.parent.parent.parent.parent))

# .env laden
try:
    from dotenv import load_dotenv
    _env_file = PROJECT_ROOT / ".env"
    if _env_file.exists():
        load_dotenv(_env_file)
    else:
        load_dotenv()
except ImportError:
    pass

from huggingface_hub import login
HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN:
    login(token=HF_TOKEN)

# vLLM Import
from vllm import LLM, SamplingParams
try:
    from vllm.sampling_params import StructuredOutputsParams
    VLLM_HAS_STRUCTURED_OUTPUTS = True
except ImportError:
    VLLM_HAS_STRUCTURED_OUTPUTS = False

# ============================================================================
# KONFIGURATION
# ============================================================================

# Failure Analysis Parameter
N_RUNS = 5             # Anzahl der Durchläufe
N_VOTING_PATHS = 1     # 1 Pfad pro Durchlauf
TEMPERATURE = 0.0      # Temperatur (0.0 für deterministisch)

BASE_MODEL_NAME = "InternVL3-14B"
MODEL_NAME = f"{BASE_MODEL_NAME}_FailureAnalysis_{N_RUNS}runs"
MODEL_HF_ID = "OpenGVLab/InternVL3-14B"

# Suchlogik für Dataset
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
SEED = 42

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Timestamped output files
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = OUTPUT_DIR / f"{MODEL_NAME}_{TIMESTAMP}_results.jsonl"
ERROR_LOG_FILE = OUTPUT_DIR / f"{MODEL_NAME}_{TIMESTAMP}_errors.jsonl"
SUMMARY_FILE = OUTPUT_DIR / f"{MODEL_NAME}_{TIMESTAMP}_summary.json"

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(OUTPUT_DIR / f"{MODEL_NAME}_{TIMESTAMP}.log")
    ]
)
logger = logging.getLogger(MODEL_NAME)

# ============================================================================
# PYDANTIC SCHEMA FÜR CoT
# ============================================================================

class AnswerChoice(str, Enum):
    A = "A"
    B = "B"
    C = "C"
    D = "D"
    E = "E"

class CoTResponse(BaseModel):
    """
    Erzwingt CoT: Erst Reasoning, dann Answer.
    """
    reasoning: str = Field(
        description="Detaillierte Schritt-für-Schritt Herleitung der Lösung auf Deutsch."
    )
    answer: AnswerChoice = Field(
        description="Der finale Lösungsbuchstabe (A, B, C, D oder E)."
    )

COT_JSON_SCHEMA = CoTResponse.model_json_schema()

# ============================================================================
# ERROR ANALYSIS
# ============================================================================

def analyze_raw_output(raw_text: str) -> Dict[str, Any]:
    """Analysiert den Raw Output des Modells um Fehlerursachen zu identifizieren."""
    analysis = {
        "raw_text": raw_text,
        "raw_length": len(raw_text) if raw_text else 0,
        "is_empty": not raw_text or raw_text.strip() == "",
        "is_ellipsis": raw_text and raw_text.strip() in ["...", "…"],
        "starts_with_brace": raw_text.strip().startswith("{") if raw_text else False,
        "ends_with_brace": raw_text.strip().endswith("}") if raw_text else False,
        "has_reasoning_key": '"reasoning"' in raw_text if raw_text else False,
        "has_answer_key": '"answer"' in raw_text if raw_text else False,
        "truncation_indicators": [],
        "json_parse_error": None,
        "pydantic_error": None,
    }
    
    if raw_text:
        if raw_text.endswith("..."):
            analysis["truncation_indicators"].append("ends_with_ellipsis")
        if not raw_text.strip().endswith("}"):
            analysis["truncation_indicators"].append("incomplete_json")
        if raw_text.count("{") != raw_text.count("}"):
            analysis["truncation_indicators"].append("unbalanced_braces")
            
        try:
            parsed = json.loads(raw_text)
            analysis["json_parse_success"] = True
            analysis["parsed_keys"] = list(parsed.keys()) if isinstance(parsed, dict) else None
            
            try:
                validated = CoTResponse(**parsed)
                analysis["pydantic_success"] = True
                analysis["answer_value"] = validated.answer.value
                analysis["reasoning_length"] = len(validated.reasoning)
            except Exception as e:
                analysis["pydantic_success"] = False
                analysis["pydantic_error"] = str(e)
                
        except json.JSONDecodeError as e:
            analysis["json_parse_success"] = False
            analysis["json_parse_error"] = {
                "message": str(e),
                "position": e.pos if hasattr(e, 'pos') else None,
            }
    
    return analysis

class ErrorCategory:
    EMPTY_OUTPUT = "empty_output"
    ELLIPSIS_ONLY = "ellipsis_only"
    TRUNCATED_JSON = "truncated_json"
    INVALID_JSON = "invalid_json"
    PYDANTIC_VALIDATION = "pydantic_validation"
    FILE_NOT_FOUND = "file_not_found"
    INFERENCE_ERROR = "inference_error"
    UNKNOWN = "unknown"

def categorize_error(analysis: Dict[str, Any], exception: Optional[Exception] = None) -> str:
    if exception:
        err_str = str(exception).lower()
        if "not found" in err_str:
            return ErrorCategory.FILE_NOT_FOUND
    
    if analysis.get("is_empty"):
        return ErrorCategory.EMPTY_OUTPUT
    if analysis.get("is_ellipsis"):
        return ErrorCategory.ELLIPSIS_ONLY
    if not analysis.get("json_parse_success"):
        if analysis.get("truncation_indicators"):
            return ErrorCategory.TRUNCATED_JSON
        return ErrorCategory.INVALID_JSON
    if not analysis.get("pydantic_success"):
        return ErrorCategory.PYDANTIC_VALIDATION
    
    return ErrorCategory.UNKNOWN

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def set_seed(seed: int):
    random.seed(seed)

def create_task_id(item: Dict) -> str:
    year = item.get('year', 'unknown')
    cls = item.get('class', 'unknown')
    task_id = item.get('task_id', 'unknown')
    
    if isinstance(cls, list):
        cls = "-".join(map(str, cls))
        
    return f"{year}_{cls}_{task_id}"

# ============================================================================
# EVALUATOR KLASSE (MIT VOTING & GENERATE FIX)
# ============================================================================

class VLMEvaluator:
    def __init__(self):
        logger.info(f"🏗️ Lade {MODEL_NAME} mit vLLM")
        logger.info(f"⚙️ Config: Failure Analysis ({N_RUNS} runs, T={TEMPERATURE})")
        
        self.error_log = []
        
        # InternVL Setup
        self.llm = LLM(
            model=MODEL_HF_ID,
            trust_remote_code=True,
            max_model_len=4096, # Ggf. auf 8192 erhöhen wenn VRAM reicht
            gpu_memory_utilization=0.9,
            dtype="bfloat16",
            limit_mm_per_prompt={"image": 1},
        )
        
        if VLLM_HAS_STRUCTURED_OUTPUTS:
            self.sampling_params = SamplingParams(
                n=N_VOTING_PATHS,            # Generiere 5 Pfade
                temperature=TEMPERATURE,     # 0.7 für Kreativität
                max_tokens=1024,             # Genug Platz für Reasoning
                structured_outputs=StructuredOutputsParams(json=COT_JSON_SCHEMA),
            )
        else:
            logger.warning("⚠️ Kein Structured Outputs - Fallback (könnte instabil sein)")
            self.sampling_params = SamplingParams(
                n=N_VOTING_PATHS,
                temperature=TEMPERATURE,
                max_tokens=1024
            )

    def generate_single_run(self, image_rel_path: str, task_id: str, run_id: int) -> Dict:
        full_path = DATA_DIR / image_rel_path
        
        result = {
            "task_id": task_id,
            "run_id": run_id,
            "image_path": image_rel_path,
            "prediction": None,
            "confidence": 0.0,
            "error_category": None,
            "error_details": None,
            "raw_output": None,
            "raw_output_analysis": None,
            "inference_time": None,
            "input_tokens": None,
            "output_tokens": None,
        }
        
        if not full_path.exists():
            result["error_category"] = ErrorCategory.FILE_NOT_FOUND
            result["error_details"] = f"File not found: {full_path}"
            logger.error(f"❌ [{task_id}][Run {run_id}] File not found: {full_path}")
            return result
        
        start_time = time.time()
        
        try:
            # 1. Bild laden
            image = Image.open(full_path).convert("RGB")
        
        # 2. Manueller Prompt-Bau (CoT angepasst)
        # Wir umgehen weiterhin das Chat-Template
        system_text = (
            "Du bist ein exzellenter Mathematik-Tutor. Deine Aufgabe ist es, Multiple-Choice-Fragen zu lösen.\n"
            "WICHTIG: Denke zuerst Schritt für Schritt nach ('reasoning'), bevor du dich auf eine Antwort festlegst."
        )
        user_text = "Analysiere das Bild und die Aufgabe. Leite die Lösung logisch her und gib am Ende die Antwort (A-E) an."
        
        # InternVL Raw Prompt Format
        prompt = f"<image>\nSystem: {system_text}\nUser: {user_text}\nAssistant:"

        inputs = {
            "prompt": prompt,
            "multi_modal_data": {"image": image},
        }

            raw_output = None
            error_msg = None
            error_traceback = None
            
            # Single run with n=1
            request_output = self.llm.generate(
                [inputs],
            if not outputs or len(outputs) == 0:
                error_msg = "Empty output from vLLM"
                return {
                    "prediction": None,
                    "raw_output": "",
                    "inference_time": duration,
                    "error": error_msg,
                    "error_traceback": None,
                    "input_tokens": input_tokens,
                    "run_id": run_id
                }
            
            raw_output = outputs[0].text.strip()
            
            # Try JSON parsing
            try:
                data = json.loads(raw_output)
                validated = CoTResponse(**data)
                return {
                    "prediction": validated.answer.value,
                    "reasoning": validated.reasoning,
                    "raw_output": raw_output,
                    "inference_time": round(duration, 4),
                    "error": None,
                    "error_traceback": None,
                    "input_tokens": input_tokens,
                    "run_id": run_id
                }
            except json.JSONDecodeError as e:
                # Fallback regex parsing
                match = re.search(r'"answer"\s*:\s*"([A-E])"', raw_output)
                if match:
                    return {
                        "prediction": match.group(1),
                        "reasoning": raw_output[:500],
                        "raw_output": raw_output,
                        "inference_time": round(duration, 4),
                        "error": f"JSON parse failed, regex fallback: {str(e)}",
                        "error_traceback": None,
                        "input_tokens": input_tokens,
                        "run_id": run_id
                    }
                else:
                    error_msg = f"JSON parse failed, no regex match: {str(e)}"
                    return {
                        "prediction": None,
                        "raw_output": raw_output,
                        "inference_time": round(duration, 4),
                        "error": error_msg,
                        "error_traceback": None,
                        "input_tokens": input_tokens,
                        "run_id": run_id
                    }
            except Exception as e:
                error_msg = f"Pydantic validation failed: {str(e)}"
                return {
                    "prediction": None,
                    "raw_output": raw_output,
                    "inference_time": round(duration, 4),
                    "error": error_msg,
                    "error_traceback": traceback.format_exc(),
                    "input_tokens": input_tokens,
                    "run_id": run_id
                }
                
        except Exception as e:
            duration = time.time() - start_time
            error_msg = f"vLLM generation failed: {str(e)}"
            error_traceback = traceback.format_exc()
            return {
                "prediction": None,
                "raw_output": raw_output or "",
                "inference_time": round(duration, 4),
                "error": error_msg,
                "error_traceback": error_traceback,
                "input_tokens": 0,
                "run_id": run_id
            }
    
    def run_multiple_passes(self, image_rel_path: str, task_id: str) -> Dict[str, Any]:
        """Run N_RUNS independent passes and collect results."""
        results = []
        for run_id in range(1, N_RUNS + 1):
            result = self.generate_single_run(image_rel_path, task_id, run_id)
            results.append(result)
        
        # Aggregate
        predictions = [r["prediction"] for r in results if r["prediction"] is not None]
        errors = [r for r in results if r["error"] is not None]
        
        if not predictions:
            return {
                "final_prediction": None,
                "confidence": 0.0,
                "all_results": results,
                "error_count": len(errors),
                "success_count": 0
            }
        
        # Majority vote
        counts = Counter(predictions)
        most_common = counts.most_common(1)[0]
        winner = most_common[0]
        confidence = most_common[1] / len(predictions)
        
        return {
            "final_prediction": winner,
            "confidence": confidence,
            "vote_distribution": dict(counts),
            "all_results": results,
            "error_count": len(errors),
            "success_count": len(predictions)
        }

    def cleanup(self):
        if hasattr(self, 'llm'):
            del self.llm
        gc.collect()

# ============================================================================
# HAUPTPROGRAMM
# ============================================================================

def run_benchmark():
    set_seed(SEED)
    
    if not DATASET_PATH.exists():
        logger.error(f"Dataset nicht gefunden: {DATASET_PATH}")
        return
        
    with open(DATASET_PATH, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    processed = set()
    if LOG_FILE.exists():
        with open(LOG_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                try: processed.add(json.loads(line).get("task_id"))
                except: pass

    logger.info(f"🚀 Starte {MODEL_NAME}: {len(dataset) - len(processed)} Aufgaben offen")
    
    evaluator = VLMEvaluator()
    correct_count = 0
    processed_count = 0
    
    try:
        with open(LOG_FILE, 'a', encoding='utf-8') as f_log:
            pbar = tqdm(dataset, desc=MODEL_NAME, unit="task")
            
            for item in pbar:
                task_id = create_task_id(item)
                
                if task_id in processed:
                    continue
                
                try:
                    image_path_raw = item.get("image_path")
                    if isinstance(image_path_raw, list):
                        image_path = image_path_raw[0]
                    else:
                        image_path = image_path_raw
                        
                    if not isinstance(image_path, str):
                        logger.warning(f"⚠️ {task_id}: Ungültiger Pfad-Typ")
                        continue

                    # Multiple runs with failure analysis
                    result = evaluator.run_multiple_passes(image_path, task_id)
                    
                    ground_truth = item.get("answer")
                    is_correct = (result["final_prediction"] == ground_truth)
                    
                    if is_correct: correct_count += 1
                    processed_count += 1
                    
                    log_entry = {
                        "model": MODEL_NAME,
                        "task_id": task_id,
                        "year": item.get("year"),
                        "class": item.get("class"),
                        "math_category": item.get("math_category"),
                        "ground_truth": ground_truth,
                        "prediction": result["final_prediction"],
                        "is_correct": is_correct,
                        "confidence": result["confidence"],
                        "vote_distribution": result.get("vote_distribution"),
                        "error_count": result["error_count"],
                        "success_count": result["success_count"],
                        "n_runs": N_RUNS
                    }
                    
                    f_log.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
                    f_log.flush()
                    
                    # Log errors to separate file
                    if result["error_count"] > 0:
                        with open(ERROR_LOG_FILE, 'a', encoding='utf-8') as f_err:
                            for run_result in result["all_results"]:
                                if run_result.get("error"):
                                    error_entry = {
                                        "task_id": task_id,
                                        "run_id": run_result["run_id"],
                                        "raw_output": run_result["raw_output"],
                                        "error": run_result["error"],
                                        "error_traceback": run_result.get("error_traceback"),
                                        "error_category": categorize_error(run_result["raw_output"], run_result["error"])
                                    }
                                    f_err.write(json.dumps(error_entry, ensure_ascii=False) + "\n")
                            f_err.flush()
                    
                    acc = correct_count / processed_count if processed_count > 0 else 0
                    pbar.set_postfix({
                        "acc": f"{acc:.1%}", 
                        "conf": f"{result['confidence']:.2f}",
                        "err": result["error_count"],
                        "last": f"{'✓' if is_correct else '✗'}"
                    })
                    
                except Exception as e:
                    logger.error(f"❌ {task_id}: {e}")
                    if "out of memory" in str(e).lower(): break
            
            pbar.close()
            
    finally:
        evaluator.cleanup()

if __name__ == "__main__":
    run_benchmark()