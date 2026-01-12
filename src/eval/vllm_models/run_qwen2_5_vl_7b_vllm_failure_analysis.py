#!/usr/bin/env python3
"""
VLM Benchmark: Känguru-Mathematik
Modell: Qwen/Qwen2.5-VL-7B-Instruct (vLLM)
Methode: Failure Analysis - Run 5 times to check consistency of null predictions

Dieses Skript führt 5 unabhängige Durchläufe durch um zu prüfen:
1. Warum manche Predictions null sind (detailliertes Error Logging)
2. Ob dieselben Items konsistent fehlschlagen (deterministisch vs. stochastisch)
"""

import os
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

# vLLM Imports
from vllm import LLM, SamplingParams
try:
    from vllm.sampling_params import StructuredOutputsParams
except ImportError:
    raise ImportError("vLLM Version zu alt. Bitte vLLM >= 0.6.0 installieren für Structured Outputs.")

# ============================================================================
# KONFIGURATION
# ============================================================================

# Analyse Parameter
N_RUNS = 5             # Anzahl der Durchläufe
N_VOTING_PATHS = 1     # 1 Pfad pro Durchlauf
TEMPERATURE = 0.0      # Temperatur (0.0 für deterministisch, >0 für Variation)

# Modell Identifikation
BASE_MODEL_NAME = "Qwen2.5-VL-7B-Instruct"
MODEL_NAME = f"{BASE_MODEL_NAME}_FailureAnalysis_{N_RUNS}runs"
MODEL_HF_ID = "Qwen/Qwen2.5-VL-7B-Instruct"

# Projekt-Setup
_script_path = Path(__file__).resolve()
PROJECT_ROOT = Path(os.environ.get("VLM_PROJECT_ROOT", _script_path.parent.parent.parent.parent))
DATASET_PATH = PROJECT_ROOT / "dataset_final.json"
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "evaluation_results"
SEED = 42

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Timestamped output files
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = OUTPUT_DIR / f"{MODEL_NAME}_{TIMESTAMP}_results.jsonl"
ERROR_LOG_FILE = OUTPUT_DIR / f"{MODEL_NAME}_{TIMESTAMP}_errors.jsonl"
SUMMARY_FILE = OUTPUT_DIR / f"{MODEL_NAME}_{TIMESTAMP}_summary.json"

# Logging Setup mit detailliertem Format
logging.basicConfig(
    level=logging.DEBUG,  # DEBUG für maximale Infos
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
        description="Detaillierte Schritt-für-Schritt Herleitung der Lösung auf Deutsch. Analysiere das Bild und den Text gründlich."
    )
    answer: AnswerChoice = Field(
        description="Der finale Lösungsbuchstabe (A, B, C, D oder E) basierend auf der Herleitung."
    )

# Schema für vLLM Guided Decoding
COT_JSON_SCHEMA = CoTResponse.model_json_schema()

# ============================================================================
# UTILS
# ============================================================================

def set_seed(seed: int):
    random.seed(seed)

def load_image_base64(image_path: Path) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def get_image_mime_type(image_path: Path) -> str:
    suffix = image_path.suffix.lower()
    return "image/png" if suffix == ".png" else "image/jpeg"

def analyze_raw_output(raw_text: str) -> Dict[str, Any]:
    """
    Analysiert den Raw Output des Modells um Fehlerursachen zu identifizieren.
    """
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
        # Check for truncation indicators
        if raw_text.endswith("..."):
            analysis["truncation_indicators"].append("ends_with_ellipsis")
        if not raw_text.strip().endswith("}"):
            analysis["truncation_indicators"].append("incomplete_json")
        if raw_text.count("{") != raw_text.count("}"):
            analysis["truncation_indicators"].append("unbalanced_braces")
        if raw_text.count('"') % 2 != 0:
            analysis["truncation_indicators"].append("unbalanced_quotes")
            
        # Try JSON parsing and capture error
        try:
            parsed = json.loads(raw_text)
            analysis["json_parse_success"] = True
            analysis["parsed_keys"] = list(parsed.keys()) if isinstance(parsed, dict) else None
            
            # Try Pydantic validation
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
                "line": e.lineno if hasattr(e, 'lineno') else None,
                "column": e.colno if hasattr(e, 'colno') else None,
            }
    
    return analysis

# ============================================================================
# ERROR CATEGORIES
# ============================================================================

class ErrorCategory:
    EMPTY_OUTPUT = "empty_output"
    ELLIPSIS_ONLY = "ellipsis_only"
    TRUNCATED_JSON = "truncated_json"
    INVALID_JSON = "invalid_json"
    PYDANTIC_VALIDATION = "pydantic_validation"
    FILE_NOT_FOUND = "file_not_found"
    INFERENCE_ERROR = "inference_error"
    OUT_OF_MEMORY = "out_of_memory"
    UNKNOWN = "unknown"

def categorize_error(analysis: Dict[str, Any], exception: Optional[Exception] = None) -> str:
    """Kategorisiert den Fehler basierend auf der Analyse."""
    if exception:
        err_str = str(exception).lower()
        if "out of memory" in err_str or "oom" in err_str:
            return ErrorCategory.OUT_OF_MEMORY
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
# EVALUATOR MIT DETAILLIERTEM ERROR LOGGING
# ============================================================================

class VLMEvaluatorWithLogging:
    def __init__(self):
        logger.info(f"🏗️ Lade {MODEL_HF_ID} mit vLLM")
        logger.info(f"⚙️ Config: Failure Analysis ({N_RUNS} runs, T={TEMPERATURE})")

        # Modell initialisieren (identisch zum funktionierenden Qwen-Script)
        self.llm = LLM(
            model=MODEL_HF_ID,
            trust_remote_code=True,
            max_model_len=16384,
            gpu_memory_utilization=0.95,
            dtype="bfloat16",
            limit_mm_per_prompt={"image": 1},
        )
        
        # Sampling Parameter
        self.sampling_params = SamplingParams(
            n=N_VOTING_PATHS,
            temperature=TEMPERATURE,
            max_tokens=1024,
            structured_outputs=StructuredOutputsParams(json=COT_JSON_SCHEMA),
        )
        
        # Error tracking
        self.error_log = []

    def generate_single_run(self, image_path: str, task_id: str, run_id: int) -> Dict:
        """
        Einzelner Inference-Durchlauf mit detailliertem Error Logging.
        """
        full_path = DATA_DIR / image_path
        
        result = {
            "task_id": task_id,
            "run_id": run_id,
            "image_path": image_path,
            "prediction": None,
            "confidence": 0.0,
            "error_category": None,
            "error_details": None,
            "raw_output": None,
            "raw_output_analysis": None,
            "inference_time": None,
            "input_tokens": None,
            "output_tokens": None,
            "finish_reason": None,
        }
        
        # Check file exists
        if not full_path.exists():
            result["error_category"] = ErrorCategory.FILE_NOT_FOUND
            result["error_details"] = f"File not found: {full_path}"
            logger.error(f"❌ [{task_id}][Run {run_id}] File not found: {full_path}")
            return result
        
        try:
            image_b64 = load_image_base64(full_path)
            mime_type = get_image_mime_type(full_path)
            
            system_prompt = (
                "Du bist ein exzellenter Mathematik-Tutor. Deine Aufgabe ist es, Multiple-Choice-Fragen zu lösen.\n"
                "WICHTIG: Denke zuerst Schritt für Schritt nach ('reasoning'), bevor du dich auf eine Antwort festlegst."
            )
            user_prompt = "Analysiere das Bild und die Aufgabe. Leite die Lösung logisch her und gib am Ende die Antwort (A-E) an."

            messages = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{image_b64}"}},
                        {"type": "text", "text": user_prompt},
                    ],
                },
            ]
            
            start_time = time.time()
            
            request_output = self.llm.chat(
                messages=messages,
                sampling_params=self.sampling_params,
                use_tqdm=False
            )
            
            duration = time.time() - start_time
            result["inference_time"] = round(duration, 4)
            
            output = request_output[0].outputs[0]
            result["raw_output"] = output.text
            result["input_tokens"] = len(request_output[0].prompt_token_ids)
            result["output_tokens"] = len(output.token_ids) if hasattr(output, 'token_ids') else None
            result["finish_reason"] = output.finish_reason if hasattr(output, 'finish_reason') else None
            
            # Detaillierte Analyse des Raw Outputs
            analysis = analyze_raw_output(output.text)
            result["raw_output_analysis"] = analysis
            
            # Versuche zu parsen
            if analysis.get("pydantic_success"):
                result["prediction"] = analysis["answer_value"]
                result["confidence"] = 1.0
                result["reasoning_length"] = analysis["reasoning_length"]
            else:
                result["error_category"] = categorize_error(analysis)
                result["error_details"] = {
                    "json_error": analysis.get("json_parse_error"),
                    "pydantic_error": analysis.get("pydantic_error"),
                    "truncation_indicators": analysis.get("truncation_indicators"),
                }
                logger.warning(
                    f"⚠️ [{task_id}][Run {run_id}] Parse failed: {result['error_category']} | "
                    f"Raw length: {analysis['raw_length']} | "
                    f"Finish reason: {result['finish_reason']} | "
                    f"Raw (first 200 chars): {output.text[:200] if output.text else 'EMPTY'}"
                )
                
        except Exception as e:
            result["error_category"] = categorize_error({}, e)
            result["error_details"] = {
                "exception_type": type(e).__name__,
                "exception_message": str(e),
                "traceback": traceback.format_exc()
            }
            logger.error(f"❌ [{task_id}][Run {run_id}] Exception: {e}")
            
        return result

    def run_multiple_passes(self, image_path: str, task_id: str) -> List[Dict]:
        """
        Führt N_RUNS unabhängige Durchläufe durch.
        """
        results = []
        for run_id in range(1, N_RUNS + 1):
            logger.debug(f"🔄 [{task_id}] Starting run {run_id}/{N_RUNS}")
            result = self.generate_single_run(image_path, task_id, run_id)
            results.append(result)
        return results

    def cleanup(self):
        if hasattr(self, 'llm'):
            del self.llm
        gc.collect()

# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def analyze_consistency(all_results: List[Dict]) -> Dict:
    """
    Analysiert die Konsistenz der Fehler über alle Items und Runs.
    """
    # Gruppiere nach task_id
    by_task = defaultdict(list)
    for r in all_results:
        by_task[r["task_id"]].append(r)
    
    analysis = {
        "total_tasks": len(by_task),
        "total_runs": len(all_results),
        "always_success": [],      # Tasks die immer erfolgreich waren
        "always_fail": [],         # Tasks die immer fehlgeschlagen sind (deterministisch)
        "intermittent": [],        # Tasks die manchmal fehlschlagen (stochastisch)
        "error_category_counts": Counter(),
        "failure_rate_by_task": {},
    }
    
    for task_id, runs in by_task.items():
        successes = sum(1 for r in runs if r["prediction"] is not None)
        failures = sum(1 for r in runs if r["prediction"] is None)
        
        failure_rate = failures / len(runs)
        analysis["failure_rate_by_task"][task_id] = {
            "successes": successes,
            "failures": failures,
            "failure_rate": failure_rate,
            "error_categories": [r.get("error_category") for r in runs if r.get("error_category")],
        }
        
        if failures == 0:
            analysis["always_success"].append(task_id)
        elif successes == 0:
            analysis["always_fail"].append(task_id)
        else:
            analysis["intermittent"].append(task_id)
        
        # Count error categories
        for r in runs:
            if r.get("error_category"):
                analysis["error_category_counts"][r["error_category"]] += 1
    
    analysis["always_success_count"] = len(analysis["always_success"])
    analysis["always_fail_count"] = len(analysis["always_fail"])
    analysis["intermittent_count"] = len(analysis["intermittent"])
    analysis["error_category_counts"] = dict(analysis["error_category_counts"])
    
    return analysis

# ============================================================================
# MAIN LOOP
# ============================================================================

def run_failure_analysis():
    set_seed(SEED)
    
    logger.info("=" * 60)
    logger.info(f"🔬 FAILURE ANALYSIS: {MODEL_NAME}")
    logger.info(f"📊 Config: {N_RUNS} runs per task, Temperature={TEMPERATURE}")
    logger.info("=" * 60)
    
    if not DATASET_PATH.exists():
        logger.error(f"Datensatz nicht gefunden: {DATASET_PATH}")
        return
        
    with open(DATASET_PATH, 'r') as f:
        dataset = json.load(f)

    logger.info(f"📁 Dataset loaded: {len(dataset)} tasks")
    
    evaluator = VLMEvaluatorWithLogging()
    all_results = []
    
    try:
        with open(LOG_FILE, 'w', encoding='utf-8') as f_log, \
             open(ERROR_LOG_FILE, 'w', encoding='utf-8') as f_error:
            
            pbar = tqdm(dataset, desc=f"{MODEL_NAME}")
            
            for item in pbar:
                task_id = f"{item.get('year')}_{item.get('class')}_{item.get('task_id')}"
                ground_truth = item.get("answer")
                
                # Run N times
                run_results = evaluator.run_multiple_passes(item["image_path"], task_id)
                
                for result in run_results:
                    result["ground_truth"] = ground_truth
                    result["is_correct"] = (result["prediction"] == ground_truth)
                    result["math_category"] = item.get("math_category")
                    result["class"] = item.get("class")
                    result["is_text_only"] = item.get("is_text_only", False)
                    
                    # Log to file
                    log_entry = {
                        "task_id": result["task_id"],
                        "run_id": result["run_id"],
                        "ground_truth": ground_truth,
                        "prediction": result["prediction"],
                        "is_correct": result["is_correct"],
                        "confidence": result["confidence"],
                        "error_category": result.get("error_category"),
                        "inference_time": result.get("inference_time"),
                        "input_tokens": result.get("input_tokens"),
                        "output_tokens": result.get("output_tokens"),
                        "finish_reason": result.get("finish_reason"),
                        "raw_output_length": len(result.get("raw_output", "") or ""),
                        "math_category": result.get("math_category"),
                        "class": result.get("class"),
                        "is_text_only": result.get("is_text_only"),
                    }
                    f_log.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
                    f_log.flush()
                    
                    # Log errors separately with full details
                    if result.get("error_category"):
                        error_entry = {
                            "task_id": result["task_id"],
                            "run_id": result["run_id"],
                            "error_category": result["error_category"],
                            "error_details": result.get("error_details"),
                            "raw_output": result.get("raw_output"),
                            "raw_output_analysis": result.get("raw_output_analysis"),
                            "image_path": result.get("image_path"),
                        }
                        f_error.write(json.dumps(error_entry, ensure_ascii=False) + "\n")
                        f_error.flush()
                    
                    all_results.append(result)
                
                # Update progress
                successes = sum(1 for r in run_results if r["prediction"] is not None)
                pbar.set_postfix({
                    "success": f"{successes}/{N_RUNS}",
                    "total": len(all_results)
                })

    except KeyboardInterrupt:
        logger.info("⚠️ Interrupted by user")
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        traceback.print_exc()
    finally:
        evaluator.cleanup()
        
        # Generate analysis summary
        logger.info("\n" + "=" * 60)
        logger.info("📊 GENERATING ANALYSIS SUMMARY")
        logger.info("=" * 60)
        
        if all_results:
            consistency_analysis = analyze_consistency(all_results)
            
            summary = {
                "model": MODEL_NAME,
                "timestamp": TIMESTAMP,
                "config": {
                    "n_runs": N_RUNS,
                    "temperature": TEMPERATURE,
                    "n_voting_paths": N_VOTING_PATHS,
                },
                "consistency_analysis": consistency_analysis,
            }
            
            with open(SUMMARY_FILE, 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            
            # Print summary
            print("\n" + "=" * 60)
            print(f"📊 FAILURE ANALYSIS SUMMARY")
            print("=" * 60)
            print(f"Total tasks: {consistency_analysis['total_tasks']}")
            print(f"Total runs: {consistency_analysis['total_runs']}")
            print(f"\n🎯 Task Categories:")
            print(f"  ✅ Always successful: {consistency_analysis['always_success_count']}")
            print(f"  ❌ Always failed (deterministic): {consistency_analysis['always_fail_count']}")
            print(f"  ⚠️ Intermittent failures (stochastic): {consistency_analysis['intermittent_count']}")
            print(f"\n📈 Error Category Distribution:")
            for cat, count in sorted(consistency_analysis['error_category_counts'].items(), key=lambda x: -x[1]):
                print(f"  {cat}: {count}")
            print(f"\n📁 Output files:")
            print(f"  Results: {LOG_FILE}")
            print(f"  Errors: {ERROR_LOG_FILE}")
            print(f"  Summary: {SUMMARY_FILE}")
            
            # List always-failing tasks
            if consistency_analysis['always_fail']:
                print(f"\n❌ Tasks that ALWAYS fail ({len(consistency_analysis['always_fail'])}):")
                for task_id in consistency_analysis['always_fail'][:20]:
                    info = consistency_analysis['failure_rate_by_task'][task_id]
                    print(f"  - {task_id}: {info['error_categories']}")
                if len(consistency_analysis['always_fail']) > 20:
                    print(f"  ... and {len(consistency_analysis['always_fail']) - 20} more")

if __name__ == "__main__":
    run_failure_analysis()
