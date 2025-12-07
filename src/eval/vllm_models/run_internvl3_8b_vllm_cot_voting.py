#!/usr/bin/env python3
"""
VLM Benchmark: Känguru-Mathematik
Modell: InternVL3-8B (vLLM Backend)
Methode: CoT + Voting (n=5) + Structured Outputs
Status: FIX für Chat-Template Bug (nutzt llm.generate)
"""

import os
import json
import logging
import re
import time
import random
import gc
import pandas as pd
from typing import Dict, List, Union
from pathlib import Path
from tqdm import tqdm
from enum import Enum
from PIL import Image
from collections import Counter

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

# Voting Parameter
N_VOTING_PATHS = 5      # 5 Pfade
TEMPERATURE = 0.7       # Diversität

MODEL_NAME = f"InternVL3-8B-CoT-Voting_n{N_VOTING_PATHS}"
MODEL_HF_ID = "OpenGVLab/InternVL3-8B"

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
        logger.info(f"⚙️ Config: CoT + Voting (k={N_VOTING_PATHS}, T={TEMPERATURE})")
        
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

    def generate_with_voting(self, image_rel_path: str) -> Dict:
        full_path = DATA_DIR / image_rel_path
        
        if not full_path.exists():
            raise FileNotFoundError(f"Bild fehlt: {full_path}")
            
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

        start_time = time.time()
        
        # 3. Generate aufrufen (liefert N Outputs zurück)
        request_output = self.llm.generate(
            [inputs],
            sampling_params=self.sampling_params,
            use_tqdm=False
        )
        
        duration = time.time() - start_time
        
        # Zugriff auf die Liste der Outputs (Größe N)
        outputs = request_output[0].outputs
        input_tokens = len(request_output[0].prompt_token_ids) if request_output[0].prompt_token_ids else 0
        
        # 4. Voting Logik
        predictions = []
        reasoning_traces = []
        parse_errors = 0

        for output in outputs:
            text = output.text.strip()
            try:
                # JSON Parsing
                data = json.loads(text)
                validated = CoTResponse(**data)
                predictions.append(validated.answer.value)
                reasoning_traces.append(validated.reasoning)
            except Exception:
                # Fallback Parsing (falls JSON Schema scheitert)
                match = re.search(r'"answer"\s*:\s*"([A-E])"', text)
                if match:
                    predictions.append(match.group(1))
                    reasoning_traces.append(text[:200] + "...")
                else:
                    parse_errors += 1
        
        if not predictions:
            return {
                "prediction": None, 
                "confidence": 0.0, 
                "inference_time": duration, 
                "error": "All paths failed parsing",
                "input_tokens": input_tokens
            }

        # Majority Vote
        counts = Counter(predictions)
        most_common = counts.most_common(1)[0]
        winner = most_common[0]
        confidence = most_common[1] / len(outputs) # z.B. 4/5 = 0.8
        
        return {
            "prediction": winner,
            "confidence": confidence,
            "vote_distribution": dict(counts),
            "reasoning_traces": reasoning_traces,
            "inference_time": round(duration, 4),
            "input_tokens": input_tokens,
            "error": None if parse_errors == 0 else f"{parse_errors} paths failed"
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

                    # Voting Aufruf
                    result = evaluator.generate_with_voting(image_path)
                    
                    ground_truth = item.get("answer")
                    is_correct = (result["prediction"] == ground_truth)
                    
                    if is_correct: correct_count += 1
                    processed_count += 1
                    
                    log_entry = {
                        "model": MODEL_NAME,
                        "task_id": task_id,
                        "year": item.get("year"),
                        "class": item.get("class"),
                        "math_category": item.get("math_category"),
                        "ground_truth": ground_truth,
                        "prediction": result["prediction"],
                        "is_correct": is_correct,
                        "confidence": result["confidence"],
                        "vote_distribution": result.get("vote_distribution"),
                        # Kurzes Reasoning für Log-File
                        "sample_reasoning": result.get("reasoning_traces", [""])[0][:500] + "...",
                        "inference_time": result["inference_time"],
                        "error_type": result["error"]
                    }
                    
                    f_log.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
                    f_log.flush()
                    
                    acc = correct_count / processed_count if processed_count > 0 else 0
                    pbar.set_postfix({
                        "acc": f"{acc:.1%}", 
                        "conf": f"{result['confidence']:.2f}",
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