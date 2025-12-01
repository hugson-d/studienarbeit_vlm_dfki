#!/usr/bin/env python3
"""
VLM Benchmark für Känguru-Mathematik-Aufgaben
Modell: InternVL3-14B
"""

import os
import json
import torch
import logging
import re
import time
import random
import gc
import pandas as pd
from PIL import Image
from typing import Dict, List, Literal, Union
from pathlib import Path
from tqdm import tqdm

from pydantic import BaseModel, ValidationError, Field
from dotenv import load_dotenv
load_dotenv()

from huggingface_hub import login
HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN:
    login(token=HF_TOKEN)

from transformers import (
    AutoProcessor, 
    AutoModelForCausalLM, 
    AutoModelForVision2Seq,
    BitsAndBytesConfig
)

# ============================================================================
# KONFIGURATION - DIESES MODELL
# ============================================================================

MODEL_NAME = "InternVL3-14B"
MODEL_HF_ID = "OpenGVLab/InternVL3-14B"
MODEL_PARAMS_B = 14
MODEL_ARCH = "internvl"

# Quantisierung ab 40B Parameter
QUANT_THRESHOLD_B = 40
USE_QUANTIZATION = MODEL_PARAMS_B > QUANT_THRESHOLD_B

# ============================================================================
# PFADE
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATASET_PATH = PROJECT_ROOT / "dataset_final.json"
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "evaluation_results"
LOG_FILE = OUTPUT_DIR / f"{MODEL_NAME}_results.jsonl"
EXCEL_FILE = OUTPUT_DIR / f"{MODEL_NAME}_summary.xlsx"
SEED = 42

# ============================================================================
# PYDANTIC SCHEMA
# ============================================================================

class KanguruAnswer(BaseModel):
    """Antwort-Schema. Wird nach Generierung validiert."""
    answer: Literal['A', 'B', 'C', 'D', 'E'] = Field(
        description="Die korrekte Antwort: A, B, C, D oder E"
    )

# ============================================================================
# SETUP
# ============================================================================

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


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def free_gpu_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    time.sleep(1)


# ============================================================================
# ANTWORT-PARSING MIT PYDANTIC
# ============================================================================

def parse_response(output_text: str) -> Dict[str, Union[str, bool, None]]:
    """Extrahiert und validiert die Antwort."""
    clean_text = output_text.strip()
    
    # Markdown Code-Blöcke entfernen
    if "```" in clean_text:
        match = re.search(r'```(?:json)?\s*(.*?)\s*```', clean_text, re.DOTALL)
        if match:
            clean_text = match.group(1).strip()
    
    # JSON-Objekt extrahieren
    json_match = re.search(r'\{[^{}]*\}', clean_text, re.DOTALL)
    
    if json_match:
        try:
            data = json.loads(json_match.group(0))
            validated = KanguruAnswer(**data)
            return {"prediction": validated.answer, "format_valid": True, "error": None}
        except json.JSONDecodeError:
            pass
        except ValidationError as e:
            error_msg = e.errors()[0].get('msg', str(e)) if e.errors() else str(e)
            return {"prediction": None, "format_valid": False, "error": f"Schema violation: {error_msg}"}
    
    # Fallback: Buchstaben-Patterns
    letter_patterns = [
        r'(?:antwort|answer|lösung|solution|ergebnis|result)[:\s]+([A-E])\b',
        r'\b([A-E])\s*(?:ist|is)\s+(?:richtig|correct|die\s+antwort|the\s+answer)',
        r'(?:^|\s)([A-E])(?:\s*$|\s*[.!])',
        r'"answer"\s*:\s*"?([A-E])"?',
    ]
    
    for pattern in letter_patterns:
        match = re.search(pattern, clean_text, re.IGNORECASE)
        if match:
            return {"prediction": match.group(1).upper(), "format_valid": False, "error": "Extracted from text"}
    
    # Letzter Fallback
    single_letter = re.search(r'\b([A-E])\b', clean_text.upper())
    if single_letter:
        return {"prediction": single_letter.group(1), "format_valid": False, "error": "Fallback: First A-E"}
    
    return {"prediction": None, "format_valid": False, "error": "No valid answer found"}


# ============================================================================
# EVALUATOR
# ============================================================================

class VLMEvaluator:
    def __init__(self):
        logger.info(f"🏗️ Lade {MODEL_NAME} ({MODEL_PARAMS_B}B)")
        logger.info(f"   HuggingFace ID: {MODEL_HF_ID}")
        logger.info(f"   4-Bit Quantisierung: {USE_QUANTIZATION}")

        bnb_config = None
        if USE_QUANTIZATION:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True, 
                bnb_4bit_quant_type="nf4", 
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True
            )
            logger.info("   📦 BitsAndBytes 4-Bit aktiviert")

        # Processor laden
        logger.info("   📥 Lade Processor...")
        try:
            self.processor = AutoProcessor.from_pretrained(MODEL_HF_ID, trust_remote_code=True)
        except Exception as e:
            logger.warning(f"   ⚠️ Standard-Processor fehlgeschlagen: {e}")
            self.processor = AutoProcessor.from_pretrained(
                MODEL_HF_ID, trust_remote_code=True,
                min_pixels=256*28*28, max_pixels=1280*28*28
            )

        # Modell-Konfiguration
        load_kwargs = {
            "device_map": "auto",
            "trust_remote_code": True,
            "torch_dtype": torch.bfloat16,
        }
        
        if bnb_config:
            load_kwargs["quantization_config"] = bnb_config
        
        # Flash Attention
        try:
            import flash_attn
            load_kwargs["attn_implementation"] = "flash_attention_2"
            logger.info("   ⚡ Flash Attention 2 aktiviert")
        except ImportError:
            load_kwargs["attn_implementation"] = "eager"

        # Modell laden
        logger.info("   📥 Lade Modell...")
        if MODEL_ARCH == "qwen":
            from transformers import Qwen2VLForConditionalGeneration
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(MODEL_HF_ID, **load_kwargs)
        elif MODEL_ARCH == "ovis":
            self.model = AutoModelForCausalLM.from_pretrained(MODEL_HF_ID, **load_kwargs)
        else:  # internvl
            self.model = AutoModelForVision2Seq.from_pretrained(MODEL_HF_ID, **load_kwargs)
        
        self.model.eval()
        logger.info(f"✅ {MODEL_NAME} bereit auf {self.model.device}")

    def generate(self, image_path: str) -> Dict:
        full_path = DATA_DIR / image_path
        if not full_path.exists():
            raise FileNotFoundError(f"Bild nicht gefunden: {full_path}")
        
        image = Image.open(full_path).convert("RGB")
        
        system_prompt = (
            "Du bist ein präzises mathematisches Assistenzsystem. "
            "Analysiere die Aufgabe im Bild und gib die korrekte Antwort. "
            "Antworte AUSSCHLIESSLICH mit einem JSON-Objekt im Format: "
            '{"answer": "X"} wobei X einer der Buchstaben A, B, C, D oder E ist.'
        )
        user_prompt = "Löse die Mathematik-Aufgabe im Bild. Gib nur das JSON zurück."

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": user_prompt}
            ]}
        ]

        try:
            text_prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            messages_simple = [{"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": f"{system_prompt}\n\n{user_prompt}"}
            ]}]
            try:
                text_prompt = self.processor.apply_chat_template(messages_simple, tokenize=False, add_generation_prompt=True)
            except Exception:
                text_prompt = f"<image>\n{system_prompt}\n{user_prompt}"

        inputs = self.processor(text=[text_prompt], images=[image], padding=True, return_tensors="pt").to(self.model.device)
        
        start_time = time.time()
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs, 
                max_new_tokens=50,
                do_sample=False,
                pad_token_id=self.processor.tokenizer.pad_token_id if hasattr(self.processor, 'tokenizer') else None
            )
        duration = time.time() - start_time
        
        input_len = inputs.input_ids.shape[1]
        trimmed_ids = [out[input_len:] for out in generated_ids]
        output_text = self.processor.batch_decode(trimmed_ids, skip_special_tokens=True)[0]
        
        del inputs, generated_ids
        
        result = parse_response(output_text)
        
        return {
            "raw_output": output_text,
            "prediction": result["prediction"],
            "format_valid": result["format_valid"],
            "error": result["error"],
            "inference_time": round(duration, 4),
            "input_tokens": input_len
        }

    def cleanup(self):
        logger.info(f"🧹 Räume {MODEL_NAME} auf...")
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'processor'):
            del self.processor
        free_gpu_memory()


# ============================================================================
# HAUPTPROGRAMM
# ============================================================================

def load_dataset() -> List[Dict]:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset nicht gefunden: {DATASET_PATH}")
    with open(DATASET_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    logger.info(f"📂 Dataset geladen: {len(data)} Aufgaben")
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
    total_tasks = len(dataset)
    
    processed = get_processed_tasks()
    remaining = total_tasks - len(processed)
    
    if remaining == 0:
        logger.info(f"⏭️ Alle {total_tasks} Tasks bereits verarbeitet.")
        return
    
    logger.info(f"🚀 Starte {MODEL_NAME}: {remaining}/{total_tasks} Tasks ausstehend")
    
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
                        "raw_output": result["raw_output"],
                        "inference_time": result["inference_time"],
                        "input_tokens": result["input_tokens"]
                    }
                    
                    f_log.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
                    f_log.flush()
                    
                    acc = correct_count / processed_count if processed_count > 0 else 0
                    status = "✓" if is_correct else "✗"
                    pbar.set_postfix({"acc": f"{acc:.1%}", "last": f"{status} GT:{ground_truth} P:{result['prediction']}"} )
                    
                except FileNotFoundError:
                    logger.warning(f"⚠️ {task_id}: Bild nicht gefunden")
                except Exception as e:
                    logger.error(f"❌ {task_id}: {str(e)[:100]}")
                    if "out of memory" in str(e).lower():
                        logger.error("💥 OOM - Breche ab")
                        break
            
            pbar.close()
        
        if processed_count > 0:
            logger.info(f"📊 {MODEL_NAME}: {correct_count}/{processed_count} = {correct_count/processed_count:.1%}")
            
    finally:
        evaluator.cleanup()


def generate_report():
    if not LOG_FILE.exists():
        logger.warning("Keine Log-Datei gefunden!")
        return
    
    df = pd.read_json(LOG_FILE, lines=True)
    if df.empty:
        logger.warning("Log-Datei ist leer!")
        return
    
    df.to_excel(EXCEL_FILE, index=False)
    
    print("\n" + "="*70)
    print(f"📊 ERGEBNISSE: {MODEL_NAME}")
    print("="*70)
    
    total = len(df)
    correct = df['is_correct'].sum()
    accuracy = df['is_correct'].mean()
    format_valid = df['format_valid'].mean()
    avg_time = df['inference_time'].mean()
    
    print(f"\nGesamtergebnis:")
    print(f"  Accuracy:     {accuracy:.1%} ({correct}/{total})")
    print(f"  Valid JSON:   {format_valid:.1%}")
    print(f"  Avg Time:     {avg_time:.2f}s")
    
    if 'math_category' in df.columns:
        print("\n📐 Nach Kategorie:")
        for cat in df['math_category'].unique():
            cat_acc = df[df['math_category'] == cat]['is_correct'].mean()
            print(f"  {cat:30s} {cat_acc:.1%}")
    
    if 'class' in df.columns:
        print("\n🎓 Nach Klassenstufe:")
        for cls in sorted(df['class'].unique()):
            cls_acc = df[df['class'] == cls]['is_correct'].mean()
            print(f"  {cls:30s} {cls_acc:.1%}")
    
    print("\n" + "="*70)
    print(f"📁 Excel: {EXCEL_FILE}")
    print(f"📜 Logs:  {LOG_FILE}")


if __name__ == "__main__":
    run_benchmark()
    generate_report()
