#!/usr/bin/env python3
"""
VLM Benchmark für Känguru-Mathematik-Aufgaben
Engine: vLLM (High-Performance)
Modell: Qwen2.5-VL-3B-Instruct
Feature: Chain-of-Thought (Reasoning) + Batch Processing
"""

import os
import json
import logging
import re
import time
import pandas as pd
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Any

# vLLM Importe
from vllm import LLM, SamplingParams

# Pydantic für Validierung
from pydantic import BaseModel, ValidationError, Field

# ============================================================================
# KONFIGURATION
# ============================================================================

MODEL_HF_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
BATCH_SIZE = 8  # Erhöhe dies auf 16 oder 32, wenn du viel VRAM (24GB+) hast
GPU_UTILIZATION = 0.90  # 90% VRAM Reservierung

# Pfade ermitteln
_script_path = Path(__file__).resolve()
PROJECT_ROOT = Path(os.environ.get("VLM_PROJECT_ROOT", _script_path.parent.parent.parent))

# Fallback Suche nach Dataset
DATASET_PATH = PROJECT_ROOT / "dataset_final.json"
if not DATASET_PATH.exists():
    # Suche in Elternordnern
    _search = _script_path.parent
    for _ in range(5):
        if (_search / "dataset_final.json").exists():
            PROJECT_ROOT = _search
            DATASET_PATH = PROJECT_ROOT / "dataset_final.json"
            break
        _search = _search.parent

DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "evaluation_results"
LOG_FILE = OUTPUT_DIR / "Qwen2.5-VL-3B-vLLM_results.jsonl"
EXCEL_FILE = OUTPUT_DIR / "Qwen2.5-VL-3B-vLLM_summary.xlsx"
SEED = 42

# Logging Setup
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("vLLM-Bench")

# ============================================================================
# DATENSTRUKTUREN
# ============================================================================

class KanguruAnswer(BaseModel):
    answer: str = Field(pattern=r"^[A-E]$")

# ============================================================================
# HILFSFUNKTIONEN
# ============================================================================

def parse_response(output_text: str) -> Dict[str, Any]:
    """
    Extrahiert die Antwort aus dem Reasoning-Output.
    Erwartet JSON am Ende oder explizite Antwort.
    """
    clean_text = output_text.strip()

    # 1. Priorität: Markdown JSON Block
    json_match = re.search(r'```json\s*(\{.*?\})\s*```', clean_text, re.DOTALL)
    if not json_match:
        # 2. Priorität: Rohes JSON-Objekt (oft am Ende)
        json_match = re.search(r'(\{[\s\S]*"answer"[\s\S]*\})', clean_text, re.DOTALL)

    if json_match:
        try:
            data = json.loads(json_match.group(1))
            # Validierung
            if "answer" in data and data["answer"] in ["A", "B", "C", "D", "E"]:
                return {"prediction": data["answer"], "format_valid": True, "error": None}
        except Exception:
            pass

    # 3. Fallback: Regex auf "Answer: X" Muster (Retter in der Not)
    fallback_patterns = [
        r'"answer"\s*:\s*"([A-E])"',
        r'Antwort:\s*([A-E])',
        r'Answer:\s*([A-E])'
    ]
    for pattern in fallback_patterns:
        match = re.search(pattern, clean_text, re.IGNORECASE)
        if match:
            return {"prediction": match.group(1).upper(), "format_valid": False, "error": "Regex Rescue"}

    return {"prediction": None, "format_valid": False, "error": "Parsing failed"}

# ============================================================================
# EVALUATOR KLASSE (vLLM)
# ============================================================================

class VLMEvaluator:
    def __init__(self):
        logger.info(f"🏗️ Starte vLLM Engine: {MODEL_HF_ID}")
        
        # Initialisierung der Engine
        self.llm = LLM(
            model=MODEL_HF_ID,
            trust_remote_code=True,
            gpu_memory_utilization=GPU_UTILIZATION,
            max_model_len=4096,  # Kontext-Limit
            limit_mm_per_prompt={"image": 1},
            tensor_parallel_size=1, # Anzahl GPUs (1 für Single GPU)
            enforce_eager=True # Manchmal stabiler für VLMs
        )
        
        # Sampling: Deterministisch (Temp 0)
        self.sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=1500,    # Genug Platz für Chain-of-Thought
            stop=["<|im_end|>"] # Stop-Token
        )
        
        # System Prompt mit CoT Instruktion
        self.system_prompt = (
            "Du bist ein exzellenter Mathematiker. Deine Aufgabe ist es, die Multiple-Choice-Frage im Bild zu lösen.\n"
            "BEFOLGE DIESEN ABLAUF:\n"
            "1. <reasoning>: Analysiere das Bild, extrahiere die Werte und löse die Aufgabe Schritt für Schritt.\n"
            "2. JSON: Gib am Ende ausschließlich das JSON-Objekt mit dem Lösungsbuchstaben aus.\n\n"
            "Beispiel-Format:\n"
            "<reasoning>\n"
            "Hier steht der Rechenweg...\n"
            "</reasoning>\n"
            "```json\n"
            "{\"answer\": \"A\"}\n"
            "```"
        )

    def prepare_prompt(self, user_text: str) -> str:
        """Baut das Qwen ChatML Format manuell zusammen."""
        # Qwen2.5-VL benötigt <|vision_start|><|image_pad|><|vision_end|> für das Bild
        return (
            f"<|im_start|>system\n{self.system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{user_text}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

    def process_batch(self, tasks: List[Dict]) -> List[Dict]:
        """Verarbeitet eine Liste von Aufgaben parallel."""
        prompts = []
        multi_modal_data = []
        valid_indices = []
        results = [None] * len(tasks)

        # 1. Daten vorbereiten
        for i, task in enumerate(tasks):
            img_path = DATA_DIR / task["image_path"]
            if not img_path.exists():
                results[i] = {"error": "Image missing", "prediction": None, "format_valid": False}
                continue
            
            try:
                image = Image.open(img_path).convert("RGB")
                
                # Prompt erstellen
                prompt_text = self.prepare_prompt("Welche Antwort (A, B, C, D oder E) ist richtig?")
                
                prompts.append(prompt_text)
                multi_modal_data.append({"image": image})
                valid_indices.append(i)
                
            except Exception as e:
                results[i] = {"error": str(e), "prediction": None, "format_valid": False}

        if not prompts:
            return results

        # 2. Inferenz (Batch)
        start_time = time.time()
        outputs = self.llm.generate(
            prompts,
            sampling_params=self.sampling_params,
            multi_modal_data=multi_modal_data,
            use_tqdm=False
        )
        duration = time.time() - start_time
        avg_time = duration / len(prompts)

        # 3. Ergebnisse verarbeiten
        for idx, output in zip(valid_indices, outputs):
            generated_text = output.outputs[0].text
            parsed = parse_response(generated_text)
            
            results[idx] = {
                "prediction": parsed["prediction"],
                "format_valid": parsed["format_valid"],
                "error": parsed["error"],
                "inference_time": round(avg_time, 4),
                "input_tokens": len(output.prompt_token_ids),
                "generated_tokens": len(output.outputs[0].token_ids),
                # Optional: Raw Text speichern, um Reasoning zu sehen
                # "raw_reasoning": generated_text 
            }

        return results

# ============================================================================
# HAUPTPROGRAMM
# ============================================================================

def load_dataset() -> List[Dict]:
    with open(DATASET_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_processed_ids() -> set:
    processed = set()
    if LOG_FILE.exists():
        with open(LOG_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    processed.add(entry.get("task_id_unique"))
                except:
                    pass
    return processed

def run_benchmark():
    dataset = load_dataset()
    evaluator = VLMEvaluator()
    
    # Eindeutige IDs erstellen
    for item in dataset:
        item["task_id_unique"] = f"{item.get('year')}_{item.get('class')}_{item.get('task_id')}"

    processed_ids = get_processed_ids()
    todo_items = [x for x in dataset if x["task_id_unique"] not in processed_ids]
    
    logger.info(f"📊 Tasks Gesamt: {len(dataset)} | Bereits fertig: {len(processed_ids)} | Offen: {len(todo_items)}")
    
    if not todo_items:
        logger.info("✅ Nichts zu tun.")
        return

    # Datei zum Anhängen öffnen
    with open(LOG_FILE, 'a', encoding='utf-8') as f_log:
        
        # Batch-Schleife
        batch_pbar = tqdm(range(0, len(todo_items), BATCH_SIZE), desc="Processing Batches")
        
        for i in batch_pbar:
            batch_items = todo_items[i : i + BATCH_SIZE]
            
            # Batch an Evaluator senden
            batch_results = evaluator.process_batch(batch_items)
            
            # Ergebnisse speichern
            for item, res in zip(batch_items, batch_results):
                ground_truth = item.get("answer")
                prediction = res.get("prediction")
                is_correct = (prediction == ground_truth) if prediction else False
                
                log_entry = {
                    "model": "Qwen2.5-VL-3B-vLLM",
                    "task_id_unique": item["task_id_unique"],
                    "original_task_id": item.get("task_id"),
                    "year": item.get("year"),
                    "class": item.get("class"),
                    "math_category": item.get("math_category"),
                    "ground_truth": ground_truth,
                    "prediction": prediction,
                    "is_correct": is_correct,
                    "format_valid": res.get("format_valid"),
                    "error": res.get("error"),
                    "inference_time": res.get("inference_time"),
                    "tokens_gen": res.get("generated_tokens")
                }
                
                f_log.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
            
            f_log.flush()
            
            # Stats Update in Progress Bar
            current_acc = sum(1 for r in batch_results if r.get("prediction") == batch_items[batch_results.index(r)].get("answer")) / len(batch_items)
            batch_pbar.set_postfix({"Batch Acc": f"{current_acc:.1%}"})

    print("\n✅ Benchmark abgeschlossen.")

def generate_report():
    if not LOG_FILE.exists():
        return
    
    df = pd.read_json(LOG_FILE, lines=True)
    if df.empty:
        return
        
    print("\n" + "="*60)
    print(f"📊 REPORT: Qwen2.5-VL-3B (vLLM)")
    print("="*60)
    
    acc = df['is_correct'].mean()
    valid = df['format_valid'].mean()
    t_avg = df['inference_time'].mean()
    
    print(f"Accuracy:   {acc:.1%} ({df['is_correct'].sum()}/{len(df)})")
    print(f"Valid JSON: {valid:.1%}")
    print(f"Avg Time:   {t_avg:.3f}s / Task")
    
    df.to_excel(EXCEL_FILE, index=False)
    print(f"💾 Excel gespeichert: {EXCEL_FILE}")

if __name__ == "__main__":
    run_benchmark()
    generate_report()