#!/usr/bin/env python3
"""
VLM Benchmark für Känguru-Mathematik-Aufgaben
Modell: Qwen2.5-VL-72B-Instruct-AWQ 4bit (vLLM Backend mit Structured Outputs / JSON Schema)

Verwendet Structured Outputs für garantierte JSON-Ausgabe.
Kompatibel mit vLLM >= 0.7.x (nutzt StructuredOutputsParams / JSON Schema).
"""

import os
import json
import logging
import re
import time
import random
import gc
import base64
import pandas as pd
from PIL import Image
from typing import Dict, List, Union
from pathlib import Path
from tqdm import tqdm
from enum import Enum

from pydantic import BaseModel, ValidationError, Field

# ============================================================================
# PROJEKT-ROOT ERMITTELN
# ============================================================================

_script_path = Path(__file__).resolve()
PROJECT_ROOT = Path(os.environ.get("VLM_PROJECT_ROOT", _script_path.parent.parent.parent.parent))

try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env", override=False)
except: pass

from huggingface_hub import login
if os.getenv("HF_TOKEN"):
    login(token=os.getenv("HF_TOKEN"))

# vLLM Import
from vllm import LLM, SamplingParams

# Versuche StructuredOutputsParams zu importieren (neue vLLM API)
try:
    from vllm.sampling_params import StructuredOutputsParams
    VLLM_HAS_STRUCTURED_OUTPUTS = True
except ImportError:
    VLLM_HAS_STRUCTURED_OUTPUTS = False

MODEL_NAME = "Qwen2.5-VL-72B-vLLM"
MODEL_HF_ID = "Qwen/Qwen2.5-VL-72B-Instruct"
MODEL_PARAMS_B = 72

DATASET_PATH = PROJECT_ROOT / "dataset_final.json"
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "evaluation_results"
LOG_FILE = OUTPUT_DIR / f"{MODEL_NAME}_results.jsonl"
SEED = 42

class AnswerChoice(str, Enum):
    A = "A"
    B = "B"
    C = "C"
    D = "D"
    E = "E"

class KanguruAnswer(BaseModel):
    answer: AnswerChoice

ANSWER_JSON_SCHEMA = KanguruAnswer.model_json_schema()

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(MODEL_NAME)

def set_seed(seed: int):
    random.seed(seed)


# ============================================================================
# ANTWORT-PARSING MIT PYDANTIC
# ============================================================================

def parse_response(output_text: str) -> Dict[str, Union[str, bool, None]]:
    clean_text = output_text.strip()
    
    try:
        data = json.loads(clean_text)
        validated = KanguruAnswer(**data)
        return {"prediction": validated.answer.value, "format_valid": True, "error": None}
    except json.JSONDecodeError:
        pass
    except ValidationError as e:
        error_msg = e.errors()[0].get('msg', str(e)) if e.errors() else str(e)
        return {"prediction": None, "format_valid": False, "error": f"Schema violation: {error_msg}"}
    
    json_match = re.search(r'\{[^{}]*\}', clean_text, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group(0))
            validated = KanguruAnswer(**data)
            return {"prediction": validated.answer.value, "format_valid": True, "error": None}
        except (json.JSONDecodeError, ValidationError):
            pass
    
    for letter in ['A', 'B', 'C', 'D', 'E']:
        if letter in clean_text.upper():
            return {"prediction": letter, "format_valid": False, "error": "Fallback extraction"}
    
    return {"prediction": None, "format_valid": False, "error": "No valid answer found"}


# ============================================================================
# BILD LADEN
# ============================================================================

def load_image_base64(image_path: Path) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def get_image_mime_type(image_path: Path) -> str:
    suffix = image_path.suffix.lower()
    mime_types = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", 
                  ".gif": "image/gif", ".webp": "image/webp", ".bmp": "image/bmp"}
    return mime_types.get(suffix, "image/jpeg")


# ============================================================================
# EVALUATOR
# ============================================================================

class VLMEvaluator:
    def __init__(self):
        self.llm = LLM(
            model=MODEL_HF_ID,
            trust_remote_code=True,
            max_model_len=4096,
            gpu_memory_utilization=0.9,
            dtype="half",
            quantization="awq",
        )
        
        if VLLM_HAS_STRUCTURED_OUTPUTS:
            structured_outputs = StructuredOutputsParams(json=ANSWER_JSON_SCHEMA)
            self.sampling_params = SamplingParams(max_tokens=50, temperature=0.0, structured_outputs=structured_outputs)
        else:
            self.sampling_params = SamplingParams(max_tokens=50, temperature=0.0)
        
        logger.info(f"Model loaded: {MODEL_NAME} ({MODEL_PARAMS_B}B)")

    def generate(self, image_path: str) -> Dict:
        full_path = DATA_DIR / image_path
        if not full_path.exists():
            raise FileNotFoundError(f"Image not found: {full_path}")
        
        image_b64 = load_image_base64(full_path)
        mime_type = get_image_mime_type(full_path)
        
        system_prompt = (
            "Du bist ein mathematisches Assistenzsystem für Multiple-Choice-Aufgaben.\n"
            "Analysiere das Bild und wähle die korrekte Antwort: A, B, C, D oder E.\n\n"
            "Antworte im JSON-Format: {\"answer\": \"X\"} wobei X = A, B, C, D oder E."
        )
        user_prompt = "Bestimme die richtige Antwort. Gib deine Antwort als JSON zurück."

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{image_b64}"}},
                    {"type": "text", "text": user_prompt}
                ]
            }
        ]
        
        start_time = time.time()
        outputs = self.llm.chat(messages=messages, sampling_params=self.sampling_params)
        duration = time.time() - start_time
        
        output_text = outputs[0].outputs[0].text.strip()
        input_tokens = len(outputs[0].prompt_token_ids) if outputs[0].prompt_token_ids else 0
        result = parse_response(output_text)
        
        return {
            "prediction": result["prediction"],
            "format_valid": result["format_valid"],
            "error": result["error"],
            "inference_time": round(duration, 4),
            "input_tokens": input_tokens,
            "raw_output": output_text
        }

    def cleanup(self):
        if hasattr(self, 'llm'):
            del self.llm
        gc.collect()


# ============================================================================
# HAUPTPROGRAMM
# ============================================================================

def load_dataset() -> List[Dict]:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")
    with open(DATASET_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    logger.info(f"Dataset loaded: {len(data)} tasks")
    return data

def get_processed_tasks() -> set:
    processed = set()
    if LOG_FILE.exists():
        with open(LOG_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    processed.add(entry.get("task_id"))
                except Exception:
                    pass
    return processed

def create_task_id(item: Dict) -> str:
    return f"{item.get('year', 'unknown')}_{item.get('class', 'unknown')}_{item.get('task_id', 'unknown')}"


def run_benchmark():
    set_seed(SEED)
    dataset = load_dataset()
    total_tasks = len(dataset)
    
    processed = get_processed_tasks()
    remaining = total_tasks - len(processed)
    
    if remaining == 0:
        logger.info(f"All {total_tasks} tasks already processed.")
        return
    
    logger.info(f"Starting {MODEL_NAME}: {remaining}/{total_tasks} tasks remaining")
    
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
                    result = evaluator.generate(item["image_path"])
                    ground_truth = item.get("answer")
                    is_correct = result["prediction"] is not None and result["prediction"] == ground_truth
                    
                    if is_correct:
                        correct_count += 1
                    processed_count += 1
                    
                    log_entry = {
                        "model": MODEL_NAME,
                        "task_id": task_id,
                        "year": item.get("year"),
                        "class": item.get("class"),
                        "original_task_id": item.get("task_id"),
                        "math_category": item.get("math_category"),
                        "is_text_only": item.get("is_text_only"),
                        "ground_truth": ground_truth,
                        "prediction": result["prediction"],
                        "is_correct": is_correct,
                        "format_valid": result["format_valid"],
                        "error_type": result["error"],
                        "inference_time": result["inference_time"],
                        "input_tokens": result["input_tokens"],
                        "raw_output": result.get("raw_output", "")
                    }
                    
                    f_log.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
                    f_log.flush()
                    
                    acc = correct_count / processed_count if processed_count > 0 else 0
                    status = "✓" if is_correct else "✗"
                    pbar.set_postfix({"acc": f"{acc:.1%}", "last": f"{status} GT:{ground_truth} P:{result['prediction']}"})
                    
                except FileNotFoundError:
                    logger.warning(f"{task_id}: Image not found")
                except Exception as e:
                    logger.error(f"{task_id}: {str(e)[:200]}")
                    if "out of memory" in str(e).lower():
                        logger.error("OOM - aborting")
                        break
            
            pbar.close()
        
        if processed_count > 0:
            logger.info(f"{MODEL_NAME}: {correct_count}/{processed_count} = {correct_count/processed_count:.1%}")
            
    finally:
        evaluator.cleanup()

def generate_report():
    if not LOG_FILE.exists():
        logger.warning("No log file found")
        return
    
    df = pd.read_json(LOG_FILE, lines=True)
    if df.empty:
        logger.warning("Log file is empty")
        return
    
    print("\n" + "="*70)
    print(f"Results: {MODEL_NAME}")
    print("="*70)
    
    total = len(df)
    correct = df['is_correct'].sum()
    accuracy = df['is_correct'].mean()
    format_valid = df['format_valid'].mean()
    avg_time = df['inference_time'].mean()
    
    print(f"\nOverall:")
    print(f"  Accuracy:     {accuracy:.1%} ({correct}/{total})")
    print(f"  Valid JSON:   {format_valid:.1%}")
    print(f"  Avg Time:     {avg_time:.2f}s")
    
    if 'math_category' in df.columns:
        print("\nBy Category:")
        for cat in df['math_category'].unique():
            cat_acc = df[df['math_category'] == cat]['is_correct'].mean()
            print(f"  {cat:30s} {cat_acc:.1%}")
    
    if 'class' in df.columns:
        print("\nBy Grade:")
        for cls in sorted(df['class'].unique()):
            cls_acc = df[df['class'] == cls]['is_correct'].mean()
            print(f"  {cls:30s} {cls_acc:.1%}")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    run_benchmark()
    generate_report()
