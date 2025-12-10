#!/usr/bin/env python3
"""
VLM Benchmark für Känguru-Mathematik-Aufgaben
Modell: Pixtral-12B (Mistral API)

Verwendet die offizielle Mistral API anstelle von lokalem vLLM.
"""

import os
import json
import logging
import re
import time
import random
import base64
import pandas as pd
from typing import Dict, List, Union, Optional
from pathlib import Path
from tqdm import tqdm
from enum import Enum

from pydantic import BaseModel, ValidationError, Field

# Projekt-Root ermitteln
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

# MISTRAL API SETUP
from mistralai import Mistral

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
if not MISTRAL_API_KEY:
    # Fallback: Versuche es hartcodiert oder warne
    print("ACHTUNG: MISTRAL_API_KEY nicht in Umgebungsvariablen gefunden!")

# ============================================================================
# KONFIGURATION
# ============================================================================

# Für Bilder (VLM) muss auf der API zwingend ein Pixtral-Modell genutzt werden.
# Stand heute ist 'pixtral-12b-2409' das Standard-Modell.
MODEL_NAME = "pixtral-12b-2409" 

DATASET_PATH = PROJECT_ROOT / "dataset_final.json"
if not DATASET_PATH.exists():
    # Fallback Suche
    _search = _script_path.parent
    for _ in range(5):
        if (_search / "dataset_final.json").exists():
            PROJECT_ROOT = _search
            DATASET_PATH = PROJECT_ROOT / "dataset_final.json"
            break
        _search = _search.parent

DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "evaluation_results"
LOG_FILE = OUTPUT_DIR / f"{MODEL_NAME}_api_results.jsonl"
SEED = 42

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

# ============================================================================
# SETUP
# ============================================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(OUTPUT_DIR / f"{MODEL_NAME}_api.log")
    ]
)
logger = logging.getLogger("MistralAPI")

def set_seed(seed: int):
    random.seed(seed)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def parse_response(output_text: str) -> Dict[str, Union[str, bool, None]]:
    """Extrahiert JSON aus der API Antwort."""
    clean_text = output_text.strip()
    
    # 1. Versuch: Direktes JSON
    try:
        data = json.loads(clean_text)
        validated = KanguruAnswer(**data)
        return {"prediction": validated.answer.value, "format_valid": True, "error": None}
    except (json.JSONDecodeError, ValidationError):
        pass
    
    # 2. Versuch: JSON Markdown Block suchen ```json ... ```
    json_match = re.search(r'```json\s*(\{.*?\})\s*```', clean_text, re.DOTALL)
    if not json_match:
        # Oder einfach nur geschweifte Klammern
        json_match = re.search(r'\{[^{}]*\}', clean_text, re.DOTALL)

    if json_match:
        try:
            data = json.loads(json_match.group(1) if json_match.groups() else json_match.group(0))
            validated = KanguruAnswer(**data)
            return {"prediction": validated.answer.value, "format_valid": True, "error": None}
        except (json.JSONDecodeError, ValidationError):
            pass
    
    # 3. Fallback: Suche nach einzelnem Buchstaben
    # Vorsicht: Das ist aggressiv, aber bei Benchmarks manchmal nötig wenn das Modell plaudert.
    matches = re.findall(r'\b([A-E])\b', clean_text.upper())
    if len(matches) == 1:
         return {"prediction": matches[0], "format_valid": False, "error": "Extracted from text (heuristic)"}

    return {"prediction": None, "format_valid": False, "error": "No valid answer found"}

def load_image_base64(image_path: Path) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def get_image_mime_type(image_path: Path) -> str:
    suffix = image_path.suffix.lower()
    return {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
    }.get(suffix, "image/jpeg")

# ============================================================================
# EVALUATOR (API VERSION)
# ============================================================================

class VLMEvaluator:
    def __init__(self):
        logger.info(f"🌐 Initialisiere Mistral API Client")
        if not MISTRAL_API_KEY:
            raise ValueError("MISTRAL_API_KEY fehlt!")
            
        self.client = Mistral(api_key=MISTRAL_API_KEY)
        self.model = MODEL_NAME

    def generate(self, image_path: str) -> Dict:
        full_path = DATA_DIR / image_path
        if not full_path.exists():
            raise FileNotFoundError(f"Bild nicht gefunden: {full_path}")
        
        # Bild vorbereiten
        image_b64 = load_image_base64(full_path)
        mime_type = get_image_mime_type(full_path)
        data_url = f"data:{mime_type};base64,{image_b64}"
        
        system_prompt = (
            "Du bist ein mathematisches Assistenzsystem für Multiple-Choice-Aufgaben. "
            "Analysiere das Bild genau. Löse die Aufgabe Schritt für Schritt. "
            "Gib am Ende NUR ein valides JSON Objekt zurück in der Form: {\"answer\": \"X\"} wobei X = A, B, C, D oder E ist."
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": system_prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": data_url 
                    }
                ]
            }
        ]

        start_time = time.time()
        
        try:
            # API Call
            chat_response = self.client.chat.complete(
                model=self.model,
                messages=messages,
                response_format={"type": "json_object"}, # Erzwingt JSON Modus (Mistral Feature)
                temperature=0.0
            )
            
            generated_text = chat_response.choices[0].message.content
            input_tokens = chat_response.usage.prompt_tokens
            
        except Exception as e:
            logger.error(f"API Error: {e}")
            return {
                "prediction": None, "format_valid": False, 
                "error": str(e), "inference_time": 0, "input_tokens": 0
            }

        duration = time.time() - start_time
        
        result = parse_response(generated_text)
        
        return {
            "prediction": result["prediction"],
            "format_valid": result["format_valid"],
            "error": result["error"],
            "inference_time": round(duration, 4),
            "input_tokens": input_tokens,
            "raw_output": generated_text
        }

# ============================================================================
# MAIN LOOP (Fast identisch zum Original)
# ============================================================================

def load_dataset() -> List[Dict]:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset nicht gefunden: {DATASET_PATH}")
    with open(DATASET_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def get_processed_tasks() -> set:
    processed = set()
    if LOG_FILE.exists():
        with open(LOG_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    processed.add(entry.get("task_id"))
                except:
                    pass
    return processed

def create_task_id(item: Dict) -> str:
    year = item.get('year', 'unknown')
    cls = item.get('class', 'unknown')
    task_id = item.get('task_id', 'unknown')
    return f"{year}_{cls}_{task_id}"

def run_benchmark():
    set_seed(SEED)
    dataset = load_dataset()
    
    processed = get_processed_tasks()
    remaining_tasks = [d for d in dataset if create_task_id(d) not in processed]
    
    logger.info(f"🚀 Starte API Benchmark mit Modell: {MODEL_NAME}")
    logger.info(f"   Tasks: {len(remaining_tasks)} offen / {len(dataset)} gesamt")
    
    if not remaining_tasks:
        return

    evaluator = VLMEvaluator()
    
    correct_count = 0
    processed_count = 0
    
    with open(LOG_FILE, 'a', encoding='utf-8') as f_log:
        pbar = tqdm(remaining_tasks, desc="API Req", unit="task")
        
        for item in pbar:
            task_id = create_task_id(item)
            
            try:
                image_path = item["image_path"]
                result = evaluator.generate(image_path)
                
                # API Rate Limits beachten (optional, aber empfohlen)
                time.sleep(1.0) 

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
                pbar.set_postfix({"acc": f"{acc:.1%}", "last": f"{status}"})
                
            except Exception as e:
                logger.error(f"❌ {task_id}: {e}")

    logger.info(f"📊 Fertig. Accuracy: {correct_count}/{processed_count} ({correct_count/processed_count:.2%})")

if __name__ == "__main__":
    run_benchmark()