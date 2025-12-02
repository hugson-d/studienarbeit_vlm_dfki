#!/usr/bin/env python3
"""
VLM Benchmark für Känguru-Mathematik-Aufgaben
Modell: InternVL3-78B
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

# Projekt-Root ZUERST ermitteln (vor dotenv laden)
_script_path = Path(__file__).resolve()
PROJECT_ROOT = Path(os.environ.get("VLM_PROJECT_ROOT", _script_path.parent.parent.parent))

# .env laden (optional - HF_TOKEN kann auch aus Umgebung kommen)
try:
    from dotenv import load_dotenv
    _env_file = PROJECT_ROOT / ".env"
    if _env_file.exists():
        load_dotenv(_env_file)
        print(f"✅ .env geladen aus: {_env_file}")
    else:
        load_dotenv()  # Fallback: aktuelles Verzeichnis
except ImportError:
    print("ℹ️ python-dotenv nicht installiert - nutze Umgebungsvariablen")

# HuggingFace Login
from huggingface_hub import login
HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN:
    login(token=HF_TOKEN)
    print(f"✅ HuggingFace Login erfolgreich")
else:
    print("⚠️ HF_TOKEN nicht gesetzt - gated models werden fehlschlagen!")

from transformers import (
    AutoTokenizer, 
    AutoModel,
    BitsAndBytesConfig
)
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

# ============================================================================
# KONFIGURATION - DIESES MODELL
# ============================================================================

MODEL_NAME = "InternVL3-78B"
MODEL_HF_ID = "OpenGVLab/InternVL3-78B"
MODEL_PARAMS_B = 78
MODEL_ARCH = "internvl"

# Quantisierung ab 40B Parameter
QUANT_THRESHOLD_B = 40
USE_QUANTIZATION = MODEL_PARAMS_B > QUANT_THRESHOLD_B

# ============================================================================
# INTERNVL IMAGE PREPROCESSING (from HuggingFace docs)
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
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image_internvl(image_file, input_size=448, max_num=12):
    """Load and preprocess image for InternVL3."""
    if isinstance(image_file, str):
        image = Image.open(image_file).convert('RGB')
    else:
        image = image_file.convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(img) for img in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

# ============================================================================
# PFADE (PROJECT_ROOT bereits oben definiert)
# ============================================================================

# Validierung: dataset_final.json muss existieren
DATASET_PATH = PROJECT_ROOT / "dataset_final.json"
if not DATASET_PATH.exists():
    # Fallback: Suche nach oben bis dataset_final.json gefunden wird
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

        # Tokenizer laden (InternVL verwendet AutoTokenizer, nicht AutoProcessor)
        logger.info("   📥 Lade Tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_HF_ID, trust_remote_code=True, use_fast=False)

        # Modell-Konfiguration
        load_kwargs = {
            "device_map": "auto",
            "trust_remote_code": True,
            "torch_dtype": torch.bfloat16,
            "low_cpu_mem_usage": True,
        }
        
        if bnb_config:
            load_kwargs["quantization_config"] = bnb_config
        
        # Flash Attention - InternVL verwendet use_flash_attn Parameter
        try:
            import flash_attn
            load_kwargs["use_flash_attn"] = True
            logger.info("   ⚡ Flash Attention 2 aktiviert")
        except ImportError:
            load_kwargs["use_flash_attn"] = False

        # Modell laden
        logger.info("   📥 Lade Modell...")
        self.model = AutoModel.from_pretrained(MODEL_HF_ID, **load_kwargs).eval()
        
        logger.info(f"✅ {MODEL_NAME} bereit")

    @torch.inference_mode()  # WICHTIG: Deaktiviert Gradientenberechnung global für diese Methode
    def generate(self, image_path: str) -> Dict:
        full_path = DATA_DIR / image_path
        if not full_path.exists():
            return {"error": "Image not found", "prediction": None}

        try:
            # InternVL-spezifische Bildvorverarbeitung
            pixel_values = load_image_internvl(str(full_path), max_num=12).to(torch.bfloat16).cuda()
            
            # Prompt für InternVL (Optimiert für JSON Output)
            system_prompt = (
                "Du bist ein präzises mathematisches Assistenzsystem. "
            "Analysiere die Aufgabe im Bild und gib die korrekte Antwort. "
            "Antworte AUSSCHLIESSLICH mit einem JSON-Objekt im Format: "
            '{"answer": "X"} wobei X einer der Buchstaben A, B, C, D oder E ist.'
            )
            user_prompt = "Löse die Mathematik-Aufgabe im Bild. Gib nur das JSON zurück."
            
            question = f"<image>\n{system_prompt}\n\n{user_prompt}"
            
            # Token Count (Schätzung)
            input_tokens = len(self.tokenizer(question).input_ids)
            
            generation_config = dict(
                max_new_tokens=128, 
                do_sample=False,
                temperature=0.0
            )
            
            start_time = time.time()
            # InternVL3 verwendet model.chat() Methode
            response = self.model.chat(self.tokenizer, pixel_values, question, generation_config)
            duration = time.time() - start_time
            
            del pixel_values
            
            result = parse_response(response)
            
            return {
                "raw_output": response,
                "prediction": result["prediction"],
                "format_valid": result["format_valid"],
                "error": result["error"],
                "inference_time": round(duration, 4),
                "input_tokens": input_tokens
            }
            
        except Exception as e:
            return {"error": str(e), "prediction": None}

    def cleanup(self):
        logger.info(f"🧹 Räume {MODEL_NAME} auf...")
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'tokenizer'):
            del self.tokenizer
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
