#!/usr/bin/env python3
"""
VLM Benchmark: Känguru-Mathematik (Temperature Sweep)
Modell: AIDC-AI/Ovis2.5-2B (vLLM)
Methode: Chain-of-Thought (CoT) + Self-Consistency Voting + Structured Outputs
"""

import os
import json
import logging
import time
import random
import gc
import base64
import pandas as pd
from collections import Counter
from typing import Dict, List, Union
from pathlib import Path
from tqdm import tqdm
from enum import Enum
from pydantic import BaseModel, Field

# vLLM Imports
from vllm import LLM, SamplingParams
try:
    from vllm.sampling_params import StructuredOutputsParams
except ImportError:
    raise ImportError("vLLM Version zu alt. Bitte vLLM >= 0.6.0 installieren.")

# ============================================================================
# KONFIGURATION
# ============================================================================

# Voting Parameter
N_VOTING_PATHS = 5      # Wichtig: Bei Temp 0.0 sind alle 5 Pfade identisch (verschwendete Rechenzeit),
                        # aber für die Datenstruktur ist es einfacher, es konstant zu lassen.
                        # Ab Temp 0.2 bringt Voting massiv Punkte.

# Temperatur-Einstellungen (Optimiert für Reasoning)
# 0.0: Greedy Baseline (Konservativ)
# 0.2 - 0.8: Der Bereich, wo Self-Consistency Voting normalerweise gewinnt
# 1.0: Obergrenze für sinnvolle Mathematik
TEMPERATURE_STEPS = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2] 

# Modell Identifikation
BASE_MODEL_NAME = "Ovis2.5-9B"
MODEL_NAME = f"{BASE_MODEL_NAME}_TempSweep_0-1_Voting-n{N_VOTING_PATHS}"
MODEL_HF_ID = "AIDC-AI/Ovis2.5-9B"

# Projekt-Setup
_script_path = Path(__file__).resolve()
PROJECT_ROOT = Path(os.environ.get("VLM_PROJECT_ROOT", _script_path.parent.parent.parent.parent))
DATASET_PATH = PROJECT_ROOT / "dataset_final.json"
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "evaluation_results"
LOG_FILE = OUTPUT_DIR / f"{MODEL_NAME}_results.jsonl"
SEED = 42

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler(OUTPUT_DIR / f"{MODEL_NAME}.log")]
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

class CoTResponse(BaseModel):
    reasoning: str = Field(description="Detaillierte Schritt-für-Schritt Herleitung auf Deutsch.")
    answer: AnswerChoice = Field(description="Der finale Lösungsbuchstabe.")

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

# ============================================================================
# EVALUATOR
# ============================================================================

class VLMEvaluator:
    def __init__(self):
        logger.info(f"🏗️ Lade {MODEL_HF_ID} mit vLLM (Engine wird nur einmal initialisiert)")
        
        # Modell initialisieren (bleibt konstant für alle Temperaturen)
        self.llm = LLM(
            model=MODEL_HF_ID,
            trust_remote_code=True,
            max_model_len=8192,
            gpu_memory_utilization=0.9,
            dtype="bfloat16",
            limit_mm_per_prompt={"image": 1},
        )

    def generate_with_voting(self, image_path: str, current_temp: float) -> Dict:
        """
        Führt die Generierung mit der spezifischen Temperatur durch.
        """
        full_path = DATA_DIR / image_path
        if not full_path.exists():
            return {"error": "Image not found", "prediction": None}
        
        image_b64 = load_image_base64(full_path)
        mime_type = get_image_mime_type(full_path)
        
        # Dynamische Sampling Params basierend auf aktueller Temp
        sampling_params = SamplingParams(
            n=N_VOTING_PATHS,
            temperature=current_temp,
            max_tokens=1024,
            structured_outputs=StructuredOutputsParams(json=COT_JSON_SCHEMA),
        )

        system_prompt = (
            "Du bist ein exzellenter Mathematik-Tutor. Löse die Multiple-Choice-Frage.\n"
            "Denke zuerst Schritt für Schritt nach ('reasoning'), bevor du antwortest."
        )
        user_prompt = "Analysiere das Bild und die Aufgabe. Leite die Lösung logisch her und gib die Antwort (A-E) an."

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{image_b64}"}},
                {"type": "text", "text": user_prompt},
            ]},
        ]
        
        start_time = time.time()
        
        # Generierung
        request_output = self.llm.chat(
            messages=messages,
            sampling_params=sampling_params,
            use_tqdm=False
        )
        
        duration = time.time() - start_time
        outputs = request_output[0].outputs
        input_tokens = len(request_output[0].prompt_token_ids)

        # Voting Logic
        predictions = []
        reasoning_traces = []
        parse_errors = 0

        for output in outputs:
            try:
                data = json.loads(output.text)
                validated = CoTResponse(**data)
                predictions.append(validated.answer.value)
                reasoning_traces.append(validated.reasoning)
            except Exception:
                parse_errors += 1
        
        if not predictions:
            return {
                "prediction": None,
                "confidence": 0.0,
                "error": "All paths failed parsing",
                "inference_time": duration
            }

        counts = Counter(predictions)
        most_common = counts.most_common(1)[0]
        
        return {
            "prediction": most_common[0],
            "confidence": most_common[1] / len(outputs),
            "vote_distribution": dict(counts),
            "reasoning_traces": reasoning_traces,
            "error": None,
            "inference_time": round(duration, 4),
            "input_tokens": input_tokens
        }

    def cleanup(self):
        if hasattr(self, 'llm'):
            del self.llm
        gc.collect()

# ============================================================================
# MAIN LOOP (TEMPERATURE SWEEP)
# ============================================================================

def run_benchmark():
    set_seed(SEED)
    
    if not DATASET_PATH.exists():
        logger.error(f"Datensatz nicht gefunden: {DATASET_PATH}")
        return
        
    with open(DATASET_PATH, 'r') as f:
        dataset = json.load(f)

    # Erweiterte Resume-Logik: Speichere (task_id, temperature) Paare
    processed_combinations = set()
    if LOG_FILE.exists():
        logger.info("Lese existierende Ergebnisse für Resume...")
        with open(LOG_FILE, 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    # Wir merken uns, welche Task bei welcher Temp schon fertig ist
                    processed_combinations.add((entry["task_id"], entry["temperature"]))
                except: pass

    logger.info(f"🚀 Starte Temperature Sweep Benchmark: {TEMPERATURE_STEPS}")
    
    evaluator = VLMEvaluator()

    try:
        with open(LOG_FILE, 'a', encoding='utf-8') as f_log:
            
            # Äußere Schleife: Temperaturen
            for temp in TEMPERATURE_STEPS:
                logger.info(f"\n🌡️  STARTE DURCHLAUF: TEMPERATURE = {temp}")
                
                # Zähler für diesen Temp-Durchlauf zurücksetzen für Logging
                temp_correct = 0
                temp_processed = 0

                pbar = tqdm(dataset, desc=f"Temp {temp}")
                
                for item in pbar:
                    task_id = f"{item.get('year')}_{item.get('class')}_{item.get('task_id')}"
                    
                    # Prüfen ob genau diese Kombination schon existiert
                    if (task_id, temp) in processed_combinations:
                        continue

                    try:
                        # Übergabe der aktuellen Temperatur an den Evaluator
                        result = evaluator.generate_with_voting(item["image_path"], current_temp=temp)
                        
                        ground_truth = item.get("answer")
                        is_correct = (result["prediction"] == ground_truth)
                        
                        if is_correct: temp_correct += 1
                        temp_processed += 1
                        
                        # Log Entry MIT Temperature Feld
                        log_entry = {
                            "task_id": task_id,
                            "temperature": temp,  # <--- NEUES FELD
                            "ground_truth": ground_truth,
                            "prediction": result["prediction"],
                            "is_correct": is_correct,
                            "confidence": result["confidence"],
                            "vote_distribution": result.get("vote_distribution"),
                            # Nur Reasoning des Gewinners speichern (oder ersten Pfad) um Platz zu sparen
                            "sample_reasoning": result.get("reasoning_traces", [""])[0][:1000],
                            "inference_time": result["inference_time"],
                            "math_category": item.get("math_category"),
                            "class": item.get("class")
                        }
                        
                        f_log.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
                        f_log.flush()
                        
                        # PBar Update
                        acc = temp_correct / temp_processed if temp_processed > 0 else 0
                        pbar.set_postfix({
                            "acc": f"{acc:.1%}", 
                            "conf": f"{result['confidence']:.2f}",
                            "T": temp
                        })

                    except Exception as e:
                        logger.error(f"❌ Error {task_id} @ T={temp}: {e}")
                        if "out of memory" in str(e).lower():
                            raise e # Beende Skript bei OOM

    finally:
        evaluator.cleanup()
        generate_report()

def generate_report():
    if not LOG_FILE.exists(): return
    df = pd.read_json(LOG_FILE, lines=True)
    if df.empty: return
    
    print(f"\n📊 Report: {MODEL_NAME}")
    
    # Gruppierung nach Temperatur
    if "temperature" in df.columns:
        summary = df.groupby("temperature").agg(
            Accuracy=("is_correct", "mean"),
            Count=("is_correct", "count"),
            Avg_Conf=("confidence", "mean")
        )
        print(summary)
    else:
        print("Keine Temperatur-Daten gefunden.")

if __name__ == "__main__":
    run_benchmark()