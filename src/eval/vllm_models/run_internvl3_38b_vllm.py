#!/usr/bin/env python3
"""
VLM Benchmark für Känguru-Mathematik-Aufgaben
Modell: InternVL3-38B (vLLM Backend)
Status: FIX für "List concatenation error" und "Image path list"
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
from typing import Dict, List, Union
from pathlib import Path
from tqdm import tqdm
from enum import Enum

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

MODEL_NAME = "InternVL3-38B-vLLM"
MODEL_HF_ID = "OpenGVLab/InternVL3-38B"

# Suchlogik für Dataset, falls PROJECT_ROOT variiert
DATASET_PATH = PROJECT_ROOT / "dataset_final.json"
if not DATASET_PATH.exists():
    # Suche rekursiv nach oben
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
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(OUTPUT_DIR / f"{MODEL_NAME}.log")
    ]
)
logger = logging.getLogger(MODEL_NAME)

# ============================================================================
# PYDANTIC SCHEMA
# ============================================================================

class AnswerChoice(str, Enum):
    A = "A"
    B = "B"
    C = "C"
    D = "D"
    E = "E"

class KanguruAnswer(BaseModel):
    answer: AnswerChoice = Field(description="Die korrekte Antwort: A, B, C, D oder E")

ANSWER_JSON_SCHEMA = KanguruAnswer.model_json_schema()

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def set_seed(seed: int):
    random.seed(seed)

def load_image_base64(image_path: Path) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def get_image_mime_type(image_path: Path) -> str:
    suffix = image_path.suffix.lower()
    return {
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".png": "image/png", ".gif": "image/gif",
        ".webp": "image/webp", ".bmp": "image/bmp"
    }.get(suffix, "image/jpeg")

def parse_response(output_text: str) -> Dict[str, Union[str, bool, None]]:
    clean_text = output_text.strip()
    try:
        data = json.loads(clean_text)
        validated = KanguruAnswer(**data)
        return {"prediction": validated.answer.value, "format_valid": True, "error": None}
    except Exception as e:
        # Fallback Regex
        match = re.search(r'"answer"\s*:\s*"([A-E])"', clean_text)
        if match:
             return {"prediction": match.group(1), "format_valid": False, "error": "Regex Fallback"}
        return {"prediction": None, "format_valid": False, "error": str(e)}

# ============================================================================
# ROBUSTE ID GENERIERUNG (FIX FÜR CRASH)
# ============================================================================

def create_task_id(item: Dict) -> str:
    """Erstellt eine ID und behandelt Listen in 'class' robust."""
    year = item.get('year', 'unknown')
    cls = item.get('class', 'unknown')
    task_id = item.get('task_id', 'unknown')
    
    # FIX: Falls Klasse eine Liste ist ["11", "12"], mache daraus "11-12"
    if isinstance(cls, list):
        cls = "-".join(map(str, cls))
        
    return f"{year}_{cls}_{task_id}"

# ============================================================================
# EVALUATOR KLASSE
# ============================================================================

class VLMEvaluator:
    def __init__(self):
        logger.info(f"🏗️ Lade {MODEL_NAME} mit vLLM")
        
        self.llm = LLM(
            model=MODEL_HF_ID,
            trust_remote_code=True,
            max_model_len=4096,
            gpu_memory_utilization=0.9,
            dtype="bfloat16",
        )
        
        if VLLM_HAS_STRUCTURED_OUTPUTS:
            logger.info("   📋 Structured Outputs aktiv")
            self.sampling_params = SamplingParams(
                max_tokens=512, temperature=0.0,
                structured_outputs=StructuredOutputsParams(json=ANSWER_JSON_SCHEMA),
            )
        else:
            self.sampling_params = SamplingParams(max_tokens=512, temperature=0.0)

    def generate(self, image_rel_path: str) -> Dict:
        full_path = DATA_DIR / image_rel_path
        
        if not full_path.exists():
            raise FileNotFoundError(f"Bild fehlt: {full_path}")
            
        image_b64 = load_image_base64(full_path)
        mime_type = get_image_mime_type(full_path)
        
        messages = [
            {"role": "system", "content": "Du bist ein mathematisches Assistenzsystem für Multiple-Choice-Aufgaben.\n"
            "Analysiere das Bild und wähle die korrekte Antwort: A, B, C, D oder E.\n\n"
            "Antworte im JSON-Format: {\"answer\": \"X\"} wobei X = A, B, C, D oder E."},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{image_b64}"}},
                    {"type": "text", "text": "Bestimme die richtige Antwort. Gib deine Antwort als JSON zurück."},
                ],
            },
        ]

        start_time = time.time()
        outputs = self.llm.chat(messages=messages, sampling_params=self.sampling_params, use_tqdm=False)
        duration = time.time() - start_time
        
        generated_text = outputs[0].outputs[0].text
        input_tokens = len(outputs[0].prompt_token_ids) if outputs[0].prompt_token_ids else 0
        
        result = parse_response(generated_text)
        result.update({
            "inference_time": round(duration, 4),
            "input_tokens": input_tokens,
            "raw_output": generated_text
        })
        return result

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
    
    # Bereits verarbeitete Tasks laden
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
                # 1. FIX: Robuste ID Erstellung
                task_id = create_task_id(item)
                
                if task_id in processed:
                    continue
                
                try:
                    # 2. FIX: Robuste Bildpfad-Extraktion
                    image_path_raw = item.get("image_path")
                    if isinstance(image_path_raw, list):
                        image_path = image_path_raw[0] # Nimm erstes Element der Liste
                    else:
                        image_path = image_path_raw
                        
                    if not isinstance(image_path, str):
                        logger.warning(f"⚠️ {task_id}: Ungültiger Pfad-Typ: {type(image_path_raw)}")
                        continue

                    # Inferenz
                    result = evaluator.generate(image_path)
                    
                    ground_truth = item.get("answer")
                    is_correct = result["prediction"] == ground_truth
                    
                    if is_correct: correct_count += 1
                    processed_count += 1
                    
                    log_entry = {
                        "model": MODEL_NAME,
                        "task_id": task_id,
                        "year": item.get("year"),
                        "class": item.get("class"), # Originaldaten behalten
                        "math_category": item.get("math_category"),
                        "ground_truth": ground_truth,
                        "prediction": result["prediction"],
                        "is_correct": is_correct,
                        "inference_time": result["inference_time"],
                        "error_type": result["error"]
                    }
                    
                    f_log.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
                    f_log.flush()
                    
                    acc = correct_count / processed_count if processed_count > 0 else 0
                    pbar.set_postfix({"acc": f"{acc:.1%}"})
                    
                except Exception as e:
                    logger.error(f"❌ {task_id}: {e}")
                    # Bei OOM abbrechen, sonst weitermachen
                    if "out of memory" in str(e).lower(): break
            
            pbar.close()
            
    finally:
        evaluator.cleanup()

if __name__ == "__main__":
    run_benchmark()
