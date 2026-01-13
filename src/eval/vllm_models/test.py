#!/usr/bin/env python3
"""
Failure Analysis: InternVL3 (vLLM Backend)
- 1 Durchlauf pro Task
- Prompt identisch zu deinem ursprünglichen InternVL-Script (Raw Prompt, kein Chat-Template)
- Schreibt ALLES (prediction, raw_output, parsing/validation infos, flags) in EIN JSONL-Output
"""

import os
import sys
import json
import logging
import gc
import re
import random
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
from enum import Enum

from tqdm import tqdm
from PIL import Image
from pydantic import BaseModel, Field

from vllm import LLM, SamplingParams
try:
    from vllm.sampling_params import StructuredOutputsParams
except ImportError:
    raise ImportError("vLLM Version zu alt. Bitte vLLM >= 0.6.0 installieren.")


# ============================================================================
# OPTIONAL: Flash Attention deaktivieren (nur falls nötig)
# ============================================================================
# os.environ["VLLM_ATTENTION_BACKEND"] = "TORCH_SDPA"
# sys.modules["flash_attn"] = None
# sys.modules["flash_attn_2_cuda"] = None


# ============================================================================
# KONFIGURATION
# ============================================================================

SEED = 42
TEMPERATURE = 0.0
MAX_TOKENS = 2024

BASE_MODEL_NAME = "InternVL3-14B"
MODEL_NAME = f"{BASE_MODEL_NAME}_FailureAnalysis_1run"
MODEL_HF_ID = "OpenGVLab/InternVL3-14B"

_script_path = Path(__file__).resolve()
PROJECT_ROOT = Path(os.environ.get("VLM_PROJECT_ROOT", _script_path.parent.parent.parent.parent))

DATASET_PATH = PROJECT_ROOT / "dataset_final.json"
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "evaluation_results"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = OUTPUT_DIR / f"{MODEL_NAME}_{TIMESTAMP}_results.jsonl"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(OUTPUT_DIR / f"{MODEL_NAME}_{TIMESTAMP}.log"),
    ],
)
logger = logging.getLogger(MODEL_NAME)


# ============================================================================
# SCHEMA & UTILS
# ============================================================================

class AnswerChoice(str, Enum):
    A = "A"
    B = "B"
    C = "C"
    D = "D"
    E = "E"

class CoTResponse(BaseModel):
    reasoning: str = Field(description="Detaillierte Schritt-für-Schritt Herleitung der Lösung auf Deutsch.")
    answer: AnswerChoice = Field(description="Der finale Lösungsbuchstabe (A, B, C, D oder E).")

COT_JSON_SCHEMA = CoTResponse.model_json_schema()

class ErrorCategory(str, Enum):
    FILE_NOT_FOUND = "file_not_found"
    EMPTY_OUTPUT = "empty_output"
    INVALID_JSON = "invalid_json"
    PYDANTIC_VALIDATION = "pydantic_validation"
    UNKNOWN = "unknown"

def set_seed(seed: int) -> None:
    random.seed(seed)

def create_task_id(item: Dict[str, Any]) -> str:
    year = item.get("year", "unknown")
    cls = item.get("class", "unknown")
    task_id = item.get("task_id", "unknown")
    if isinstance(cls, list):
        cls = "-".join(map(str, cls))
    return f"{year}_{cls}_{task_id}"

def is_reasoning_placeholder(reasoning: Optional[str]) -> bool:
    """
    Erfasst Fälle wie ".." sowie unbrauchbar kurze Platzhalter.
    """
    if reasoning is None:
        return True
    r = reasoning.strip()
    if r in {".", "..", "...", "…"}:
        return True
    if len(r) <= 5 and re.fullmatch(r"[.\s…]+", r or ""):
        return True
    if len(r) < 15:
        return True
    return False

def analyze_raw_output(raw_text: Optional[str]) -> Dict[str, Any]:
    """
    Analysiert:
    - empty vs non-empty
    - json parse ok?
    - pydantic validation ok?
    - answer_value / reasoning length / reasoning placeholder
    Liefert außerdem failure_stage + failure_message für klare Diagnose.
    """
    analysis: Dict[str, Any] = {
        "raw_text_present": bool(raw_text),
        "raw_length": len(raw_text) if raw_text else 0,
        "json_parse_success": False,
        "pydantic_success": False,
        "failure_stage": None,
        "failure_message": None,
        "answer_value": None,
        "reasoning_length": None,
        "reasoning_is_placeholder": None,
        "is_empty": False,
    }

    if not raw_text or not raw_text.strip():
        analysis["is_empty"] = True
        analysis["failure_stage"] = "empty_output"
        analysis["failure_message"] = "Model returned empty/whitespace output."
        return analysis

    # JSON parse
    try:
        parsed = json.loads(raw_text)
        analysis["json_parse_success"] = True
    except Exception as e:
        analysis["failure_stage"] = "json_parse"
        analysis["failure_message"] = str(e)
        return analysis

    # Pydantic validate
    try:
        validated = CoTResponse(**parsed)
        analysis["pydantic_success"] = True
        analysis["answer_value"] = validated.answer.value
        analysis["reasoning_length"] = len(validated.reasoning or "")
        analysis["reasoning_is_placeholder"] = is_reasoning_placeholder(validated.reasoning or "")
        return analysis
    except Exception as e:
        analysis["failure_stage"] = "pydantic_validation"
        analysis["failure_message"] = str(e)
        return analysis


# ============================================================================
# EVALUATOR (InternVL: llm.generate + multi_modal_data)
# ============================================================================

class InternVLEvaluatorFailureAnalysis:
    def __init__(self):
        logger.info(f"🏗️ Initialisiere {MODEL_HF_ID} (generate-mode, kein Chat-Template)")

        self.llm = LLM(
            model=MODEL_HF_ID,
            trust_remote_code=True,
            max_model_len=8192,
            gpu_memory_utilization=0.90,
            dtype="bfloat16",
            limit_mm_per_prompt={"image": 1},
        )

        self.sampling_params = SamplingParams(
            n=1,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            structured_outputs=StructuredOutputsParams(json=COT_JSON_SCHEMA),
        )

        # PROMPT MUSS IDENTISCH BLEIBEN (wie in deinem InternVL Voting-Script)
        self.system_text = (
            "Du bist ein exzellenter Mathematik-Tutor. Deine Aufgabe ist es, Multiple-Choice-Fragen zu lösen.\n"
            "WICHTIG: Denke zuerst Schritt für Schritt nach ('reasoning'), bevor du dich auf eine Antwort festlegst."
        )
        self.user_text = (
            "Analysiere das Bild und die Aufgabe. Leite die Lösung logisch her und gib am Ende die Antwort (A-E) an."
        )

    def generate_once(self, image_rel_path: str, task_id: str) -> Dict[str, Any]:
        full_path = DATA_DIR / image_rel_path

        res: Dict[str, Any] = {
            "model": MODEL_NAME,
            "task_id": task_id,
            "image_path": image_rel_path,
            "prediction": None,
            "invalid_output": None,
            "reasoning_is_placeholder": None,
            "error_category": None,
        }

        if not full_path.exists():
            res["error_category"] = ErrorCategory.FILE_NOT_FOUND.value
            res["invalid_output"] = True
            return res

        try:
            image = Image.open(full_path).convert("RGB")

            # IDENTISCHER Raw-Prompt wie im InternVL Voting-Script
            prompt = f"<image>\nSystem: {self.system_text}\nUser: {self.user_text}\nAssistant:"

            inputs = {
                "prompt": prompt,
                "multi_modal_data": {"image": image},
            }

            request_output = self.llm.generate(
                [inputs],
                sampling_params=self.sampling_params,
                use_tqdm=False,
            )

            out0 = request_output[0].outputs[0]
            raw_text = (out0.text or "").strip()

            prompt_tokens = (
                len(request_output[0].prompt_token_ids)
                if getattr(request_output[0], "prompt_token_ids", None)
                else None
            )
            completion_tokens = (
                len(out0.token_ids)
                if getattr(out0, "token_ids", None)
                else None
            )

            analysis = analyze_raw_output(raw_text)

            res.update({
                "raw_output": raw_text,  # FULL, damit man genau sieht, was kam
                "finish_reason": getattr(out0, "finish_reason", None),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "raw_output_analysis": analysis,
            })

            # Kategorie + prediction
            if analysis.get("is_empty"):
                res["error_category"] = ErrorCategory.EMPTY_OUTPUT.value
                res["prediction"] = None
            elif analysis.get("json_parse_success") and analysis.get("pydantic_success"):
                res["error_category"] = None
                res["prediction"] = analysis.get("answer_value")
                res["reasoning_is_placeholder"] = analysis.get("reasoning_is_placeholder", False)
            elif analysis.get("json_parse_success") and not analysis.get("pydantic_success"):
                res["error_category"] = ErrorCategory.PYDANTIC_VALIDATION.value
                res["prediction"] = None
            else:
                res["error_category"] = ErrorCategory.INVALID_JSON.value
                res["prediction"] = None

            res["invalid_output"] = (res["prediction"] is None)
            if res["invalid_output"] and res.get("reasoning_is_placeholder") is None:
                # wenn kein valid JSON -> reasoning unbekannt
                res["reasoning_is_placeholder"] = None

            return res

        except Exception as e:
            res["error_category"] = ErrorCategory.UNKNOWN.value
            res["invalid_output"] = True
            res["error_details"] = str(e)
            res["traceback"] = traceback.format_exc()
            return res

    def cleanup(self):
        if hasattr(self, "llm"):
            del self.llm
        gc.collect()


# ============================================================================
# MAIN
# ============================================================================

def run_failure_analysis_internvl():
    set_seed(SEED)

    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset nicht gefunden: {DATASET_PATH}")

    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    evaluator = InternVLEvaluatorFailureAnalysis()

    invalid_count = 0
    placeholder_reasoning_count = 0

    with open(LOG_FILE, "w", encoding="utf-8") as f_log:
        for item in tqdm(dataset, desc=MODEL_NAME, unit="task"):
            task_id = create_task_id(item)

            image_path_raw = item.get("image_path")
            image_path = image_path_raw[0] if isinstance(image_path_raw, list) else image_path_raw

            # Dataset-Fehler auch sauber als Datensatz loggen
            if not isinstance(image_path, str):
                r = {
                    "model": MODEL_NAME,
                    "task_id": task_id,
                    "year": item.get("year"),
                    "class": item.get("class"),
                    "math_category": item.get("math_category"),
                    "ground_truth": item.get("answer"),
                    "image_path": image_path_raw,
                    "prediction": None,
                    "raw_output": None,
                    "finish_reason": None,
                    "prompt_tokens": None,
                    "completion_tokens": None,
                    "raw_output_analysis": {
                        "raw_text_present": False,
                        "raw_length": 0,
                        "json_parse_success": False,
                        "pydantic_success": False,
                        "failure_stage": "dataset",
                        "failure_message": f"Ungültiger image_path-Typ: {type(image_path_raw)}",
                        "answer_value": None,
                        "reasoning_length": None,
                        "reasoning_is_placeholder": None,
                        "is_empty": True,
                    },
                    "error_category": ErrorCategory.UNKNOWN.value,
                    "invalid_output": True,
                    "reasoning_is_placeholder": None,
                }
                invalid_count += 1
                f_log.write(json.dumps(r, ensure_ascii=False) + "\n")
                continue

            r = evaluator.generate_once(image_path, task_id)

            # Dataset-Metadaten in denselben Datensatz schreiben (alles in EINEM JSON)
            r["ground_truth"] = item.get("answer")
            r["math_category"] = item.get("math_category")
            r["year"] = item.get("year")
            r["class"] = item.get("class")

            if r.get("invalid_output"):
                invalid_count += 1
            if r.get("reasoning_is_placeholder") is True:
                placeholder_reasoning_count += 1

            f_log.write(json.dumps(r, ensure_ascii=False) + "\n")

    evaluator.cleanup()

    logger.info(f"Fertig. Log: {LOG_FILE}")
    logger.info(f"Counts: invalid_prediction={invalid_count}, reasoning_placeholder={placeholder_reasoning_count}")


if __name__ == "__main__":
    run_failure_analysis_internvl()
