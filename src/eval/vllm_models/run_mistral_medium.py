#!/usr/bin/env python3
"""
VLM Benchmark für Känguru-Mathematik-Aufgaben
Modell: Mistral Large / Pixtral (Mistral API)
Feature: Native Structured Outputs (JSON Schema Enforcement)
"""

import os
import json
import logging
import time
import random
import base64
import re
from pathlib import Path
from typing import Dict, List, Union, Optional
from enum import Enum
from tqdm import tqdm

from pydantic import BaseModel, ValidationError, Field

# MISTRAL API SDK
from mistralai import Mistral

# ============================================================================
# SETUP & KONFIGURATION
# ============================================================================

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

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
if not MISTRAL_API_KEY:
    print("⚠️ ACHTUNG: MISTRAL_API_KEY nicht gefunden!")

# Modell Wahl
# "mistral-large-2512" (Mistral Large 2) oder "pixtral-12b-2409"
MODEL_NAME = "mistral-medium-2508" 

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
LOG_FILE = OUTPUT_DIR / f"{MODEL_NAME}_structured_api_results.jsonl"
SEED = 42

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Logging Setup
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
# 1. PYDANTIC SCHEMA DEFINITION
# ============================================================================

class AnswerChoice(str, Enum):
    A = "A"
    B = "B"
    C = "C"
    D = "D"
    E = "E"

class KanguruAnswer(BaseModel):
    """
    Strukturiertes Format für die Antwort.
    Das Feld 'reasoning' hilft dem Modell, logisch zu denken, bevor es antwortet (CoT).
    """
    reasoning: str = Field(
        description="Schritt-für-Schritt Lösung der Aufgabe auf Deutsch. Analysiere das Bild und den Text."
    )
    answer: AnswerChoice = Field(
        description="Der finale Lösungsbuchstabe: A, B, C, D oder E"
    )

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

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
# EVALUATOR KLASSE
# ============================================================================

class VLMEvaluator:
    def __init__(self):
        logger.info(f"🌐 Initialisiere Mistral API Client für {MODEL_NAME}")
        self.client = Mistral(api_key=MISTRAL_API_KEY)
        self.model = MODEL_NAME

    def generate(self, image_path: str) -> Dict:
        full_path = DATA_DIR / image_path
        if not full_path.exists():
            return {"prediction": None, "error": "Image not found", "format_valid": False}
        
        # 1. Bild vorbereiten
        image_b64 = load_image_base64(full_path)
        mime_type = get_image_mime_type(full_path)
        data_url = f"data:{mime_type};base64,{image_b64}"
        
        # 2. Prompt erstellen
        system_prompt = "Du bist ein mathematisches Assistenzsystem für Multiple-Choice-Aufgaben.\n" \
            "Analysiere das Bild und wähle die korrekte Antwort: A, B, C, D oder E.\n\n" \
            "Antworte im JSON-Format: {\"answer\": \"X\"} wobei X = A, B, C, D oder E."
        user_prompt = "Bestimme die richtige Antwort. Gib deine Antwort als JSON zurück."
        
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": data_url},
                    {"type": "text", "text": user_prompt}
                ]
            }
        ]

        # 3. API Call mit STRUCTURED OUTPUTS
        start_time = time.time()
        
        try:
            chat_response = self.client.chat.complete(
                model=self.model,
                messages=messages,
                # HIER IST DIE MAGIC: Wir übergeben das Pydantic Schema
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "math_solution",  # Ein Name für das Schema ist Pflicht
                        "schema": KanguruAnswer.model_json_schema(), # Das Schema als Dict
                        "strict": True            # Erzwingt strikte Einhaltung
                    }
                },
                temperature=0.0
            )
            
            # Da 'strict=True', ist der Content garantiert valides JSON passend zum Schema
            generated_text = chat_response.choices[0].message.content
            input_tokens = chat_response.usage.prompt_tokens
            
            # Parsen
            data = json.loads(generated_text)
            validated_obj = KanguruAnswer(**data)
            
            prediction = validated_obj.answer.value
            reasoning = validated_obj.reasoning
            format_valid = True
            error = None

        except Exception as e:
            logger.error(f"API/Parse Error: {e}")
            return {
                "prediction": None,
                "reasoning": None,
                "format_valid": False,
                "error": str(e),
                "inference_time": 0,
                "input_tokens": 0,
                "raw_output": str(e)
            }

        duration = time.time() - start_time
        
        return {
            "prediction": prediction,
            "reasoning": reasoning,
            "format_valid": format_valid,
            "error": error,
            "inference_time": round(duration, 4),
            "input_tokens": input_tokens,
            "raw_output": generated_text
        }

# ============================================================================
# MAIN LOOP
# ============================================================================

def load_dataset() -> List[Dict]:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset nicht gefunden: {DATASET_PATH}")
    with open(DATASET_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_processed_ids() -> set:
    processed = set()
    if LOG_FILE.exists():
        with open(LOG_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                try: processed.add(json.loads(line).get("task_id"))
                except: pass
    return processed

def create_task_id(item: Dict) -> str:
    return f"{item.get('year')}_{item.get('class')}_{item.get('task_id')}"

def run_benchmark():
    set_seed(SEED)
    dataset = load_dataset()
    processed = get_processed_ids()
    
    # Filtern, was noch zu tun ist
    tasks_to_do = [d for d in dataset if create_task_id(d) not in processed]
    
    logger.info(f"🚀 Starte Structured API Benchmark: {MODEL_NAME}")
    logger.info(f"   Schema: KanguruAnswer (Reasoning + Answer)")
    logger.info(f"   Tasks: {len(tasks_to_do)} offen")

    if not tasks_to_do: return

    evaluator = VLMEvaluator()
    correct_count = 0
    processed_count = 0
    
    with open(LOG_FILE, 'a', encoding='utf-8') as f_log:
        pbar = tqdm(tasks_to_do, desc="API Req", unit="task")
        
        for item in pbar:
            task_id = create_task_id(item)
            
            try:
                # API Call
                result = evaluator.generate(item["image_path"])
                
                # Pause für Rate Limits (wichtig bei großen Modellen wie Mistral Large)
                time.sleep(1.0) 

                ground_truth = item.get("answer")
                is_correct = result["prediction"] == ground_truth
                
                if is_correct: correct_count += 1
                processed_count += 1
                
                log_entry = {
                    "model": MODEL_NAME,
                    "task_id": task_id,
                    "year": item.get("year"),
                    "class": item.get("class"),
                    "ground_truth": ground_truth,
                    "prediction": result["prediction"],
                    "reasoning": result.get("reasoning", "")[:500] + "...", # Gekürzt für Log
                    "is_correct": is_correct,
                    "format_valid": result["format_valid"],
                    "error": result["error"],
                    "inference_time": result["inference_time"],
                    "input_tokens": result["input_tokens"]
                }
                
                f_log.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
                f_log.flush()
                
                acc = correct_count / processed_count
                status = "✓" if is_correct else "✗"
                pbar.set_postfix({"acc": f"{acc:.1%}", "last": f"{status}"})
                
            except Exception as e:
                logger.error(f"❌ {task_id}: {e}")

    logger.info(f"📊 Final: {correct_count}/{processed_count} ({correct_count/processed_count:.1%})")

if __name__ == "__main__":
    run_benchmark()
    
    