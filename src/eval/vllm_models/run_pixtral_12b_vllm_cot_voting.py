#!/usr/bin/env python3
"""
VLM Benchmark: Känguru-Mathematik
Modell: pixtral-12b-2409 (Mistral API)
Methode: Chain-of-Thought (CoT) + Self-Consistency Voting (Majority Vote)
"""

import os
import json
import logging
import time
import random
import base64
import re
import pandas as pd
from collections import Counter
from typing import Dict, List, Union, Optional
from pathlib import Path
from tqdm import tqdm
from enum import Enum
from pydantic import BaseModel, Field, ValidationError

# MISTRAL API
from mistralai import Mistral

# Projekt-Setup
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
    print("⚠️ ACHTUNG: MISTRAL_API_KEY fehlt!")

# ============================================================================
# KONFIGURATION
# ============================================================================

# Voting Parameter
N_VOTING_PATHS = 5      # Wie viele Versuche pro Bild? (Kostet mehr Credits!)
TEMPERATURE = 0.7       # Etwas höher für Diversität beim Voting

# Modell
MODEL_NAME = "pixtral-12b-2409"
BENCHMARK_NAME = f"Pixtral-API_CoT-Voting_n{N_VOTING_PATHS}"

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
LOG_FILE = OUTPUT_DIR / f"{BENCHMARK_NAME}_results.jsonl"
SEED = 42

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler(OUTPUT_DIR / f"{BENCHMARK_NAME}.log")]
)
logger = logging.getLogger("MistralVoting")

# ============================================================================
# PYDANTIC SCHEMA
# ============================================================================

class AnswerChoice(str, Enum):
    A = "A"
    B = "B"
    C = "C"
    D = "D"
    E = "E"

class CoTResponse(BaseModel):
    reasoning: str = Field(description="Schritt-für-Schritt Lösung")
    answer: AnswerChoice = Field(description="Der finale Lösungsbuchstabe")

# ============================================================================
# HELPER
# ============================================================================

def load_image_base64(image_path: Path) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def get_image_mime_type(image_path: Path) -> str:
    suffix = image_path.suffix.lower()
    return "image/png" if suffix == ".png" else "image/jpeg"

def parse_response(text: str) -> Optional[Dict]:
    """Versucht JSON aus dem Text zu extrahieren."""
    clean = text.strip()
    # 1. Direkt JSON
    try:
        data = json.loads(clean)
        return CoTResponse(**data).dict()
    except: pass
    
    # 2. Markdown JSON
    match = re.search(r'```json\s*(\{.*?\})\s*```', clean, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(1))
            return CoTResponse(**data).dict()
        except: pass

    # 3. Rohe Klammern
    match = re.search(r'\{.*\}', clean, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            return CoTResponse(**data).dict()
        except: pass
        
    return None

# ============================================================================
# EVALUATOR
# ============================================================================

class VLMEvaluator:
    def __init__(self):
        self.client = Mistral(api_key=MISTRAL_API_KEY)
        logger.info(f"🤖 Pixtral API Voting System (n={N_VOTING_PATHS}, T={TEMPERATURE})")

    def _single_request(self, messages) -> Optional[Dict]:
        """Führt einen einzelnen API Call durch."""
        try:
            response = self.client.chat.complete(
                model=MODEL_NAME,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=TEMPERATURE
            )
            content = response.choices[0].message.content
            return parse_response(content)
        except Exception as e:
            logger.warning(f"API Request failed: {e}")
            return None

    def generate_with_voting(self, image_path: str) -> Dict:
        full_path = DATA_DIR / image_path
        if not full_path.exists():
            return {"error": "Image not found", "prediction": None}

        # Bild laden
        b64 = load_image_base64(full_path)
        mime = get_image_mime_type(full_path)
        url = f"data:{mime};base64,{b64}"

        # System Prompt für CoT (Konsistent mit anderen Skripten)
        system_prompt = (
            "Du bist ein exzellenter Mathematik-Tutor. Deine Aufgabe ist es, Multiple-Choice-Fragen zu lösen.\n"
            "WICHTIG: Denke zuerst Schritt für Schritt nach ('reasoning'), bevor du dich auf eine Antwort festlegst."
        )

        # User Prompt mit Bild
        user_prompt = "Analysiere das Bild und die Aufgabe. Leite die Lösung logisch her und gib am Ende die Antwort (A-E) an."

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": url},
                    {"type": "text", "text": user_prompt}
                ]
            }
        ]

        start_time = time.time()
        
        # Sequentiell N Calls (keine Parallelität)
        results = []
        for i in range(N_VOTING_PATHS):
            res = self._single_request(messages)
            if res:
                results.append(res)
            # Kurze Pause zwischen Calls um Rate Limits zu schonen
            time.sleep(0.5)
        
        duration = time.time() - start_time

        # Voting Auswertung
        valid_answers = [r['answer'] for r in results if r and r.get('answer')]
        
        if not valid_answers:
            return {
                "prediction": None,
                "confidence": 0.0,
                "error": "All API calls failed or invalid JSON",
                "inference_time": duration
            }

        # Majority Vote
        counts = Counter(valid_answers)
        most_common = counts.most_common(1)[0]
        winner = most_common[0]
        confidence = most_common[1] / len(valid_answers) # Confidence relativ zu validen Antworten

        # Bestes Reasoning finden (das zum Gewinner gehört)
        winner_reasoning = next((r['reasoning'] for r in results if r and r['answer'] == winner), "")

        return {
            "prediction": winner,
            "confidence": confidence,
            "vote_distribution": dict(counts),
            "reasoning_sample": winner_reasoning,
            "total_calls": N_VOTING_PATHS,
            "successful_calls": len(valid_answers),
            "inference_time": round(duration, 4)
        }

# ============================================================================
# MAIN
# ============================================================================

def run_benchmark():
    if not DATASET_PATH.exists():
        logger.error("Dataset missing")
        return

    with open(DATASET_PATH, 'r') as f:
        dataset = json.load(f)

    # Bereits verarbeitete Aufgaben filtern
    processed_ids = set()
    if LOG_FILE.exists():
        with open(LOG_FILE, 'r') as f:
            for line in f:
                try:
                    processed_ids.add(json.loads(line)['task_id'])
                except: pass
    
    tasks_to_do = [d for d in dataset if f"{d.get('year')}_{d.get('class')}_{d.get('task_id')}" not in processed_ids]
    
    logger.info(f"🚀 Start API Voting Benchmark. Tasks: {len(tasks_to_do)}")
    
    evaluator = VLMEvaluator()
    correct = 0
    count = 0
    
    with open(LOG_FILE, 'a', encoding='utf-8') as f_log:
        pbar = tqdm(tasks_to_do, desc="Pixtral Voting")
        
        for item in pbar:
            task_id = f"{item.get('year')}_{item.get('class')}_{item.get('task_id')}"
            
            try:
                res = evaluator.generate_with_voting(item["image_path"])
                
                gt = item.get("answer")
                is_correct = (res["prediction"] == gt)
                
                if is_correct: correct += 1
                count += 1
                
                log_entry = {
                    "task_id": task_id,
                    "ground_truth": gt,
                    "prediction": res["prediction"],
                    "is_correct": is_correct,
                    "confidence": res.get("confidence", 0),
                    "vote_distribution": res.get("vote_distribution"),
                    "sample_reasoning": res.get("reasoning_traces", [""])[0][:500] + "...",
                    "inference_time": res.get("inference_time"),
                    "class": item.get("class"),
                    "category": item.get("math_category")
                }
                
                f_log.write(json.dumps(log_entry) + "\n")
                f_log.flush()
                
                pbar.set_postfix({
                    "acc": f"{correct/count:.1%}",
                    "conf": f"{res.get('confidence',0):.2f}"
                })
                
            except Exception as e:
                logger.error(f"Error at {task_id}: {e}")

    # Report
    if count > 0:
        print(f"\n✅ Fertig! Accuracy: {correct/count:.1%}")

if __name__ == "__main__":
    run_benchmark()