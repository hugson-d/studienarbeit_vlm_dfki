#!/usr/bin/env python3
"""
VLM Benchmark für Känguru-Mathematik-Aufgaben
Modell: InternVL3-8B (Optimized)
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

# HuggingFace & Torch
from huggingface_hub import login
from transformers import (
    AutoTokenizer, 
    AutoModel,
    BitsAndBytesConfig
)
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

# ============================================================================
# SETUP & KONFIGURATION
# ============================================================================

# Pfad-Setup
_script_path = Path(__file__).resolve()
PROJECT_ROOT = Path(os.environ.get("VLM_PROJECT_ROOT", _script_path.parent.parent.parent))
DATASET_PATH = PROJECT_ROOT / "dataset_final.json"
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "evaluation_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "InternVL3-8B"
MODEL_HF_ID = "OpenGVLab/InternVL3-8B"
SEED = 42

# Logging Setup
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(OUTPUT_DIR / f"{MODEL_NAME}.log")
    ]
)
logger = logging.getLogger(MODEL_NAME)

# Token Handling
HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN:
    login(token=HF_TOKEN)
else:
    logger.warning("⚠️ HF_TOKEN nicht gesetzt.")

# ============================================================================
# INTERNVL UTILITIES (Boilerplate for Image Preprocessing)
# ============================================================================

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) 
        for i in range(1, n + 1) for j in range(1, n + 1) 
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        processed_images.append(resized_img.crop(box))
    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))
    return processed_images

def load_image_internvl(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(img) for img in images]
    return torch.stack(pixel_values)

# ============================================================================
# EVALUATOR CLASS
# ============================================================================

class KanguruAnswer(BaseModel):
    answer: Literal['A', 'B', 'C', 'D', 'E'] = Field(description="Die korrekte Antwort.")

class VLMEvaluator:
    def __init__(self):
        logger.info(f"🏗️ Initialisiere {MODEL_NAME}...")
        
        # 1. Konfiguration für Memory/Speed
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # InternVL3 ist groß, wir nutzen bfloat16
        torch_dtype = torch.bfloat16
        
        load_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch_dtype,
            "low_cpu_mem_usage": True,
            "device_map": "auto"
        }

        # 2. Flash Attention 2 Integration (Wichtig für Speed/Memory bei VLM)
        try:
            import flash_attn
            load_kwargs["attn_implementation"] = "flash_attention_2"
            logger.info("   ⚡ Flash Attention 2 aktiviert")
        except ImportError:
            logger.info("   ⚠️ Flash Attention 2 nicht verfügbar, nutze Fallback.")

        # 3. Tokenizer laden
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_HF_ID, trust_remote_code=True, use_fast=False
        )

        # 4. Modell laden (Kritischer Fix: AutoModel statt AutoModelForVision2Seq)
        try:
            self.model = AutoModel.from_pretrained(MODEL_HF_ID, **load_kwargs).eval()
        except Exception as e:
            logger.error(f"FATAL: Modell konnte nicht geladen werden. Fehler: {e}")
            raise e

        logger.info(f"✅ Modell geladen auf {self.device}")

    @torch.inference_mode()  # WICHTIG: Deaktiviert Gradientenberechnung global für diese Methode
    def generate(self, image_path: str) -> Dict:
        full_path = DATA_DIR / image_path
        if not full_path.exists():
            return {"error": "Image not found", "prediction": None}

        try:
            # Preprocessing
            pixel_values = load_image_internvl(str(full_path), max_num=12).to(torch.bfloat16).cuda()
            
            # Prompt Engineering
            question = (
                "<image>\n"
                "Löse die folgende Mathematik-Aufgabe. Analysiere das Bild genau.\n"
                "Gib die Antwort NUR als JSON-Objekt im Format {\"answer\": \"X\"}, "
                "wobei X einer der Buchstaben A, B, C, D oder E ist."
            )
            
            generation_config = dict(
                max_new_tokens=128, 
                do_sample=False,
                temperature=0.0
            )

            start_time = time.time()
            # Model.chat ist spezifisch für InternVL Remote Code
            response = self.model.chat(self.tokenizer, pixel_values, question, generation_config)
            duration = time.time() - start_time

            result = parse_response(response)
            
            return {
                "raw_output": response,
                "prediction": result["prediction"],
                "format_valid": result["format_valid"],
                "error": result["error"],
                "inference_time": round(duration, 4)
            }
            
        except Exception as e:
            return {"error": str(e), "prediction": None}

    def cleanup(self):
        del self.model
        del self.tokenizer
        gc.collect()
        torch.cuda.empty_cache()

# ============================================================================
# PARSING & UTILS
# ============================================================================

def parse_response(output_text: str) -> Dict:
    clean_text = output_text.strip()
    # Versuch 1: JSON Parsing
    try:
        # Finde JSON Block
        match = re.search(r'\{.*?\}', clean_text, re.DOTALL)
        if match:
            data = json.loads(match.group(0))
            # Normalisiere Keys (manchmal generieren LLMs 'Answer' statt 'answer')
            data = {k.lower(): v for k, v in data.items()}
            if 'answer' in data and data['answer'] in ['A', 'B', 'C', 'D', 'E']:
                return {"prediction": data['answer'], "format_valid": True, "error": None}
    except:
        pass

    # Versuch 2: Regex Fallback
    patterns = [
        r'Antwort:\s*([A-E])',
        r'Answer:\s*([A-E])',
        r'The correct answer is\s*([A-E])',
        r'^\s*([A-E])\s*$'
    ]
    for p in patterns:
        m = re.search(p, clean_text, re.IGNORECASE | re.MULTILINE)
        if m:
            return {"prediction": m.group(1).upper(), "format_valid": False, "error": "Regex Extraction"}

    return {"prediction": None, "format_valid": False, "error": "No valid answer"}

def run_benchmark():
    # Setup
    random.seed(SEED)
    torch.manual_seed(SEED)
    
    # Dataset laden
    if not DATASET_PATH.exists():
        logger.error(f"Dataset nicht gefunden: {DATASET_PATH}")
        return
    with open(DATASET_PATH) as f:
        dataset = json.load(f)

    # Schon bearbeitete filtern
    processed_ids = set()
    if (OUTPUT_DIR / f"{MODEL_NAME}_results.jsonl").exists():
        with open(OUTPUT_DIR / f"{MODEL_NAME}_results.jsonl", 'r') as f:
            for line in f:
                try: 
                    processed_ids.add(json.loads(line)['task_id']) 
                except: pass

    # Evaluator starten
    evaluator = VLMEvaluator()
    
    pbar = tqdm(dataset, desc="Processing")
    log_file = OUTPUT_DIR / f"{MODEL_NAME}_results.jsonl"
    
    for task in pbar:
        task_id = f"{task.get('year')}_{task.get('class')}_{task.get('task_id')}"
        
        if task_id in processed_ids:
            continue
            
        result = evaluator.generate(task.get("image_path"))
        
        is_correct = (result['prediction'] == task.get('answer')) if result['prediction'] else False
        
        log_entry = {
            "task_id": task_id,
            "original_id": task.get("task_id"),
            "answer_gt": task.get("answer"),
            "answer_pred": result["prediction"],
            "is_correct": is_correct,
            **result
        }
        
        with open(log_file, 'a') as f:
            f.write(json.dumps(log_entry) + "\n")

    evaluator.cleanup()
    logger.info("Benchmark beendet.")

if __name__ == "__main__":
    run_benchmark()