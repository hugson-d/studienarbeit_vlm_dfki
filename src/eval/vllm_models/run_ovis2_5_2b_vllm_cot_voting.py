#!/usr/bin/env python3
"""
VLM Benchmark: Känguru-Mathematik
Modell: AIDC-AI/Ovis2.5-2B (vLLM)
Methode: Chain-of-Thought (CoT) + Self-Consistency Voting (Majority Vote) + Structured Outputs
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
    raise ImportError("vLLM Version zu alt. Bitte vLLM >= 0.6.0 installieren für Structured Outputs.")

# ============================================================================
# KONFIGURATION
# ============================================================================

# Voting Parameter
N_VOTING_PATHS = 5      # 5 Pfade (Sweetspot für Benchmark)
TEMPERATURE = 0.7       # Temperatur > 0 für Diversität

# Modell Identifikation
BASE_MODEL_NAME = "Ovis2.5-2B"
MODEL_NAME = f"{BASE_MODEL_NAME}_CoT-Voting_n{N_VOTING_PATHS}"
MODEL_HF_ID = "AIDC-AI/Ovis2.5-2B"

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

# ============================================================================
# EVALUATOR MIT VOTING
# ============================================================================

class VLMEvaluator:
    def __init__(self):
        logger.info(f"🏗️ Lade {MODEL_HF_ID} mit vLLM")
        logger.info(f"⚙️ Config: CoT + Voting (k={N_VOTING_PATHS}, T={TEMPERATURE})")

        # Modell initialisieren
        # Ovis2.5 benötigt limit_mm_per_prompt und ausreichend Context
        self.llm = LLM(
            model=MODEL_HF_ID,
            trust_remote_code=True,
            max_model_len=8192,   # 8k reicht für Ovis 2B gut aus
            gpu_memory_utilization=0.9,
            dtype="bfloat16",
            limit_mm_per_prompt={"image": 1},
        )
        
        # Sampling Parameter für Voting
        self.sampling_params = SamplingParams(
            n=N_VOTING_PATHS,            # Generiere N unabhängige Antworten
            temperature=TEMPERATURE,     # Diversität
            max_tokens=1024,             # Genug Platz für Reasoning
            structured_outputs=StructuredOutputsParams(json=COT_JSON_SCHEMA),
        )

    def generate_with_voting(self, image_path: str) -> Dict:
        full_path = DATA_DIR / image_path
        if not full_path.exists():
            raise FileNotFoundError(f"Bild nicht gefunden: {full_path}")
        
        image_b64 = load_image_base64(full_path)
        mime_type = get_image_mime_type(full_path)
        
        # System Prompt für CoT (Identisch zum Qwen-Setup für Vergleichbarkeit)
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
                    {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{image_b64}"}},
                    {"type": "text", "text": user_prompt},
                ],
            },
        ]
        
        start_time = time.time()
        
        # Generierung (1 Aufruf liefert N Pfade)
        request_output = self.llm.chat(
            messages=messages,
            sampling_params=self.sampling_params,
            use_tqdm=False
        )
        
        duration = time.time() - start_time
        outputs = request_output[0].outputs  # Liste von N CompletionOutput Objekten
        input_tokens = len(request_output[0].prompt_token_ids)

        # --- Voting Logic ---
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
        
        # Ergebnis ermitteln
        if not predictions:
            return {
                "prediction": None,
                "confidence": 0.0,
                "error": "All paths failed parsing",
                "inference_time": duration,
                "input_tokens": input_tokens
            }

        # Majority Vote
        counts = Counter(predictions)
        most_common = counts.most_common(1)[0]
        winner_answer = most_common[0]
        votes = most_common[1]
        confidence = votes / len(outputs)

        return {
            "prediction": winner_answer,
            "confidence": confidence,
            "vote_distribution": dict(counts),
            "reasoning_traces": reasoning_traces, 
            "error": None if parse_errors == 0 else f"{parse_errors} paths failed",
            "inference_time": round(duration, 4),
            "input_tokens": input_tokens
        }

    def cleanup(self):
        if hasattr(self, 'llm'):
            del self.llm
        gc.collect()

# ============================================================================
# MAIN LOOP
# ============================================================================

def run_benchmark():
    set_seed(SEED)
    
    if not DATASET_PATH.exists():
        logger.error(f"Datensatz nicht gefunden: {DATASET_PATH}")
        return
        
    with open(DATASET_PATH, 'r') as f:
        dataset = json.load(f)

    # Resume-Logik
    processed = set()
    if LOG_FILE.exists():
        with open(LOG_FILE, 'r') as f:
            for line in f:
                try:
                    processed.add(json.loads(line)["task_id"])
                except: pass

    logger.info(f"🚀 Starte Benchmark: {len(dataset) - len(processed)} Aufgaben verbleibend")
    
    evaluator = VLMEvaluator()
    correct_count = 0
    processed_count = 0

    try:
        with open(LOG_FILE, 'a', encoding='utf-8') as f_log:
            pbar = tqdm(dataset, desc=MODEL_NAME)
            
            for item in pbar:
                task_id = f"{item.get('year')}_{item.get('class')}_{item.get('task_id')}"
                
                if task_id in processed:
                    continue

                try:
                    result = evaluator.generate_with_voting(item["image_path"])
                    
                    ground_truth = item.get("answer")
                    is_correct = (result["prediction"] == ground_truth)
                    
                    if is_correct:
                        correct_count += 1
                    processed_count += 1
                    
                    # Log Entry - Reasoning gekürzt für Log
                    log_entry = {
                        "task_id": task_id,
                        "ground_truth": ground_truth,
                        "prediction": result["prediction"],
                        "is_correct": is_correct,
                        "confidence": result["confidence"],
                        "vote_distribution": result.get("vote_distribution"),
                        "sample_reasoning": result.get("reasoning_traces", [""])[0][:500] + "...",
                        "inference_time": result["inference_time"],
                        "math_category": item.get("math_category"),
                        "class": item.get("class")
                    }
                    
                    f_log.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
                    f_log.flush()
                    
                    acc = correct_count / processed_count
                    pbar.set_postfix({
                        "acc": f"{acc:.1%}", 
                        "conf": f"{result['confidence']:.2f}",
                        "last": f"{'✓' if is_correct else '✗'}"
                    })

                except Exception as e:
                    logger.error(f"❌ Error {task_id}: {e}")
                    if "out of memory" in str(e).lower():
                        break

    finally:
        evaluator.cleanup()
        generate_report()

def generate_report():
    if not LOG_FILE.exists(): return
    df = pd.read_json(LOG_FILE, lines=True)
    if df.empty: return
    
    print(f"\n📊 Report: {MODEL_NAME}")
    print(f"Accuracy: {df['is_correct'].mean():.1%} ({df['is_correct'].sum()}/{len(df)})")
    print(f"Avg Confidence: {df['confidence'].mean():.2f}")
    print(f"Avg Time: {df['inference_time'].mean():.2f}s")

if __name__ == "__main__":
    run_benchmark()