#!/usr/bin/env python3
"""
VLM Benchmark: Känguru-Mathematik
Modell: HuggingFaceM4/Idefics3-8B-Llama3 (vLLM)
Methode: CoT + Voting (n=5) + Structured Outputs + HF-Chat-Template
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
from PIL import Image  # NEU: für Bildübergabe an vLLM

# vLLM Imports
from vllm import LLM, SamplingParams
try:
    from vllm.sampling_params import StructuredOutputsParams
    VLLM_HAS_STRUCTURED_OUTPUTS = True
except ImportError:
    VLLM_HAS_STRUCTURED_OUTPUTS = False
    raise ImportError("vLLM Version zu alt oder Structured Outputs nicht verfügbar.")

# HF Processor für Chat-Template
from transformers import AutoProcessor

# ============================================================================
# KONFIGURATION
# ============================================================================

# Voting Parameter
N_VOTING_PATHS = 1      # 5 Pfade
TEMPERATURE = 0.0       # Diversität

MODEL_NAME = f"Idefics3-8B-CoT-Voting_n{N_VOTING_PATHS}"
MODEL_HF_ID = "HuggingFaceM4/Idefics3-8B-Llama3"

# Projekt-Setup
_script_path = Path(__file__).resolve()
PROJECT_ROOT = Path(os.environ.get("VLM_PROJECT_ROOT", _script_path.parent.parent.parent.parent))
DATASET_PATH = PROJECT_ROOT / "dataset_final.json"
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
        description="Detaillierte Schritt-für-Schritt Herleitung der Lösung auf Deutsch. Analysiere das Bild und den Text gründlich."
    )
    answer: AnswerChoice = Field(
        description="Der finale Lösungsbuchstabe (A, B, C, D oder E) basierend auf der Herleitung."
    )

COT_JSON_SCHEMA = CoTResponse.model_json_schema()

# ============================================================================
# UTILS
# ============================================================================

def set_seed(seed: int):
    random.seed(seed)

# (load_image_base64 / get_image_mime_type bleiben ungenutzt, können aber stehen bleiben,
# falls du sie später noch brauchst)
def load_image_base64(image_path: Path) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def get_image_mime_type(image_path: Path) -> str:
    suffix = image_path.suffix.lower()
    return "image/png" if suffix == ".png" else "image/jpeg"

# ============================================================================
# EVALUATOR MIT VOTING + CHAT-TEMPLATE
# ============================================================================

class VLMEvaluator:
    def __init__(self):
        logger.info(f"🏗️ Lade {MODEL_HF_ID} mit vLLM")
        logger.info(f"⚙️ Config: CoT + Voting (k={N_VOTING_PATHS}, T={TEMPERATURE})")
        logger.info(f"📋 Structured Outputs mit Schema: reasoning + answer (A–E)")

        # 1) HF-Processor nur für Chat-Template
        self.processor = AutoProcessor.from_pretrained(MODEL_HF_ID)

        # 2) vLLM-LMM (Idefics3)
        self.llm = LLM(
            model=MODEL_HF_ID,
            trust_remote_code=True,
            max_model_len=8192,  # genug Platz für Bild-Patches + CoT
            gpu_memory_utilization=0.9,
            dtype="bfloat16",
            limit_mm_per_prompt={"image": 1},
            # optional: Bild-Preprocessing/Größe feintunen
            mm_processor_kwargs={
                "size": {
                    "longest_edge": 3 * 364
                }
            },
        )
        
        # 3) Sampling für Voting + Structured Outputs
        self.sampling_params = SamplingParams(
            n=N_VOTING_PATHS,                        # 5 Pfade
            temperature=TEMPERATURE,                 # Diversität
            max_tokens=1024,                         # Platz für Reasoning
            structured_outputs=StructuredOutputsParams(json=COT_JSON_SCHEMA),
        )

    def generate_with_voting(self, image_path: str) -> Dict:
        full_path = DATA_DIR / image_path
        if not full_path.exists():
            raise FileNotFoundError(f"Bild nicht gefunden: {full_path}")
        
        # Bild als PIL-Image (vLLM kümmert sich um Vorverarbeitung)
        image = Image.open(full_path).convert("RGB")
        
        # System-Prompt: CoT + JSON-Schema klarmachen
        system_prompt = (
            "Du bist ein exzellenter Mathematik-Tutor. Deine Aufgabe ist es, Multiple-Choice-Fragen zu lösen.\n"
            "WICHTIG: Denke zuerst Schritt für Schritt nach ('reasoning'), bevor du dich auf eine Antwort festlegst."
        )
        
        user_prompt = (
            "Hier ist die Känguru-Mathematik-Aufgabe:\n"
            "<image>\n\n"
            "Bestimme die korrekte Antwort basierend auf dem Bild. Gib nur das JSON zurück."
        )

        # Messages im HF-Idefics3-Format (für apply_chat_template)
        messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": user_prompt}
                ]
            }
        ]

        # Chat-Template in einen Prompt-String umwandeln
        prompt = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False
        )
        
        start_time = time.time()
        
        # vLLM-Generierung: ein Request mit n=5 Outputs
        request_output = self.llm.generate(
            {
                "prompt": prompt,
                "multi_modal_data": {
                    "image": image
                },
            },
            sampling_params=self.sampling_params,
            use_tqdm=False
        )
        
        duration = time.time() - start_time
        outputs = request_output[0].outputs
        input_tokens = len(request_output[0].prompt_token_ids) if request_output[0].prompt_token_ids else 0
        
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
                "vote_distribution": {},
                "reasoning_traces": [],
                "error": f"All {len(outputs)} paths failed parsing",
                "inference_time": round(duration, 4),
                "input_tokens": input_tokens
            }

        # Majority Vote
        counts = Counter(predictions)
        most_common = counts.most_common(1)[0]
        winner = most_common[0]
        confidence = most_common[1] / len(outputs)
        
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
        if hasattr(self, 'processor'):
            del self.processor
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
                try:
                    processed.add(json.loads(line).get("task_id"))
                except:
                    pass

    logger.info(f"🚀 Starte {MODEL_NAME}: {len(dataset) - len(processed)} Aufgaben offen")
    
    evaluator = VLMEvaluator()
    correct_count = 0
    processed_count = 0
    
    try:
        with open(LOG_FILE, 'a', encoding='utf-8') as f_log:
            pbar = tqdm(dataset, desc=MODEL_NAME, unit="task")
            
            for item in pbar:
                # Task ID bauen (je nach Struktur im JSON)
                year = item.get('year', 'unknown')
                cls = item.get('class', 'unknown')
                t_id = item.get('task_id', 'unknown')
                task_id = f"{year}_{cls}_{t_id}"
                
                if task_id in processed:
                    continue
                
                try:
                    image_path_raw = item.get("image_path")
                    # Fallback falls List
                    if isinstance(image_path_raw, list):
                        image_path = image_path_raw[0]
                    else:
                        image_path = image_path_raw

                    result = evaluator.generate_with_voting(image_path)
                    
                    ground_truth = item.get("answer")
                    is_correct = (result["prediction"] == ground_truth)
                    
                    if is_correct:
                        correct_count += 1
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
                        # Gekürztes Reasoning speichern
                        "sample_reasoning": (
                            result.get("reasoning_traces", [""])[0][:500] + "..."
                            if result.get("reasoning_traces") else ""
                        ),
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
                    if "out of memory" in str(e).lower():
                        logger.error("💥 OOM - Abbruch.")
                        break
            
            pbar.close()
            
    finally:
        evaluator.cleanup()

if __name__ == "__main__":
    run_benchmark()
