#!/usr/bin/env python3
"""
VLM Benchmark für Känguru-Mathematik-Aufgaben
Modell: InternVL3-8B
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
from typing import Dict, Literal
from pathlib import Path
from tqdm import tqdm
from pydantic import BaseModel, ValidationError, Field

# HuggingFace & Torch
from huggingface_hub import login
from transformers import (
    AutoTokenizer,
    AutoModel,
    BitsAndBytesConfig,  # falls du später 4-bit/8-bit testen willst
)
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

# ============================================================================
# SETUP & KONFIGURATION
# ============================================================================

_script_path = Path(__file__).resolve()
PROJECT_ROOT = Path(os.environ.get("VLM_PROJECT_ROOT", _script_path.parent.parent.parent))
DATASET_PATH = PROJECT_ROOT / "dataset_final.json"
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "evaluation_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "InternVL3-8B"
MODEL_HF_ID = "OpenGVLab/InternVL3-8B"
LOG_FILE = OUTPUT_DIR / f"{MODEL_NAME}_results.jsonl"
EXCEL_FILE = OUTPUT_DIR / f"{MODEL_NAME}_summary.xlsx"
SEED = 42

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(OUTPUT_DIR / f"{MODEL_NAME}.log"),
    ],
)
logger = logging.getLogger(MODEL_NAME)

HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN:
    login(token=HF_TOKEN)

# ============================================================================
# INTERNVL IMAGE PREPROCESSING (Dynamic Tiling)
# ============================================================================

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size: int):
    transform = T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
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
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
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
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        processed_images.append(resized_img.crop(box))

    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))

    return processed_images


def load_image_internvl(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert("RGB")
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(
        image, image_size=input_size, use_thumbnail=True, max_num=max_num
    )
    pixel_values = [transform(img) for img in images]
    return torch.stack(pixel_values)  # (num_tiles, 3, H, W)


# ============================================================================
# PARSING & UTILS
# ============================================================================


class KanguruAnswer(BaseModel):
    answer: Literal["A", "B", "C", "D", "E"] = Field(description="Die korrekte Antwort.")


def parse_response(output_text: str) -> Dict:
    clean_text = output_text.strip()

    # Code-Block entfernen, falls vorhanden
    if "```" in clean_text:
        match = re.search(r"```(?:json)?\s*(.*?)\s*```", clean_text, re.DOTALL)
        if match:
            clean_text = match.group(1).strip()

    # JSON-Suche
    json_match = re.search(r"\{[^{}]*\}", clean_text, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group(0))
            data = {k.lower(): v for k, v in data.items()}
            validated = KanguruAnswer(**data)
            return {
                "prediction": validated.answer,
                "format_valid": True,
                "error": None,
            }
        except (json.JSONDecodeError, ValidationError):
            pass

    # Regex-Fallbacks
    patterns = [
        r"(?:antwort|answer|lösung|solution)[:\s]+([A-E])\b",
        r'"answer"\s*:\s*"?([A-E])"?',
    ]
    for p in patterns:
        m = re.search(p, clean_text, re.IGNORECASE)
        if m:
            return {
                "prediction": m.group(1).upper(),
                "format_valid": False,
                "error": "Regex Extraction",
            }

    # Letzter Fallback: letztes A–E im Text
    last_letter_match = re.findall(r"\b([A-E])\b", clean_text.upper())
    if last_letter_match:
        return {
            "prediction": last_letter_match[-1],
            "format_valid": False,
            "error": "Fallback: Last A-E",
        }

    return {"prediction": None, "format_valid": False, "error": "No valid answer"}


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================================
# EVALUATOR
# ============================================================================


class VLMEvaluator:
    def __init__(self):
        logger.info("Initialisiere %s ...", MODEL_NAME)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Einfaches, robustes Lade-Setup; keine exotischen attn-Flags
        load_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.bfloat16,
            "low_cpu_mem_usage": True,
        }

        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_HF_ID,
            trust_remote_code=True,
            use_fast=False,
        )

        self.model = AutoModel.from_pretrained(
            MODEL_HF_ID,
            attn_implementation="eager",  # Flash Attention deaktiviert (Container hat alte Version)
            **load_kwargs,
        ).eval()

        if self.device.type == "cuda":
            self.model.cuda()

        logger.info("Modell geladen. Device: %s", self.device)

    @torch.inference_mode()
    def generate(self, image_path: str) -> Dict:
        full_path = DATA_DIR / image_path
        if not full_path.exists():
            return {"error": f"Image not found: {full_path}", "prediction": None}

        try:
            # Bild laden und auf dasselbe Device wie das Modell verschieben
            pixel_values = load_image_internvl(str(full_path), max_num=12)
            model_device = next(self.model.parameters()).device
            pixel_values = pixel_values.to(model_device, dtype=torch.bfloat16)

            # Prompting (wie bei dir / Gemma / Qwen)
            system_prompt = (
                "Du bist ein mathematisches Assistenzsystem für Multiple-Choice-Aufgaben.\n\n"
                "AUFGABE: Analysiere das Bild und wähle die korrekte Antwort.\n\n"
                "ZWINGENDE AUSGABE - NUR DIESES FORMAT IST ERLAUBT:\n"
                '{"answer": "X"}\n'
                "wobei X EXAKT einer dieser Buchstaben sein MUSS: A, B, C, D oder E\n\n"
                "WICHTIG:\n"
                "- Deine GESAMTE Antwort besteht NUR aus diesem JSON-Objekt.\n"
                "- KEINE anderen Zeichen, Wörter oder Erklärungen.\n"
                "- Bei Unsicherheit: Wähle die wahrscheinlichste Option (A-E).\n"
                "- Eine Antwort ist PFLICHT - du musst A, B, C, D oder E wählen."
            )
            user_prompt = (
                "Bestimme die korrekte Antwort basierend auf dem Bild. Gib nur das JSON zurück."
            )

            # InternVL-Chat-Format
            question = f"<image>\n{system_prompt}\n\n{user_prompt}"

            # grobe Tokenanzahl
            input_tokens = len(self.tokenizer(question).input_ids)

            generation_config = dict(
                max_new_tokens=128,
                do_sample=False,
                temperature=0.0,
            )

            start_time = time.time()
            # Remote-Code-Chat-Interface von InternVL
            response = self.model.chat(
                self.tokenizer,
                pixel_values,
                question,
                generation_config,
            )
            duration = time.time() - start_time

            result = parse_response(response)

            return {
                "prediction": result["prediction"],
                "format_valid": result["format_valid"],
                "error": result["error"],
                "inference_time": round(duration, 4),
                "input_tokens": input_tokens,
            }

        except Exception as e:
            return {"error": str(e), "prediction": None}

    def cleanup(self):
        del self.model
        del self.tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ============================================================================
# MAIN LOOP
# ============================================================================


def run_benchmark():
    set_seed(SEED)

    if not DATASET_PATH.exists():
        logger.error("Dataset fehlt: %s", DATASET_PATH)
        return

    with open(DATASET_PATH) as f:
        dataset = json.load(f)

    # Resume-Logik
    processed_ids = set()
    if LOG_FILE.exists():
        with open(LOG_FILE, "r") as f:
            for line in f:
                try:
                    processed_ids.add(json.loads(line)["task_id"])
                except Exception:
                    pass

    evaluator = VLMEvaluator()

    correct_count = 0
    processed_count = 0

    with open(LOG_FILE, "a") as f_log:
        pbar = tqdm(dataset, desc=MODEL_NAME)
        for task in pbar:
            task_id = f"{task.get('year')}_{task.get('class')}_{task.get('task_id')}"

            if task_id in processed_ids:
                continue

            result = evaluator.generate(task.get("image_path"))

            gt = task.get("answer")
            is_correct = (result["prediction"] == gt) if result["prediction"] else False

            if is_correct:
                correct_count += 1
            processed_count += 1

            log_entry = {
                "model": MODEL_NAME,
                "task_id": task_id,
                "year": task.get("year"),
                "class": task.get("class"),
                "original_task_id": task.get("task_id"),
                "math_category": task.get("math_category"),
                "is_text_only": task.get("is_text_only", False),
                "ground_truth": gt,
                "prediction": result["prediction"],
                "is_correct": is_correct,
                "format_valid": result.get("format_valid"),
                "error_type": result.get("error"),
                "inference_time": result.get("inference_time"),
                "input_tokens": result.get("input_tokens"),
            }

            f_log.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
            f_log.flush()

            acc = correct_count / processed_count if processed_count > 0 else 0.0
            pbar.set_postfix({"acc": f"{acc:.1%}"})

    evaluator.cleanup()
    generate_report()


def generate_report():
    if not LOG_FILE.exists():
        return
    df = pd.read_json(LOG_FILE, lines=True)
    if df.empty:
        return

    df.to_excel(EXCEL_FILE, index=False)

    print("\n" + "=" * 70)
    print(f"ERGEBNISSE: {MODEL_NAME}")
    print(f"  Accuracy:   {df['is_correct'].mean():.1%}")
    print(f"  Valid JSON: {df['format_valid'].mean():.1%}")

    if "math_category" in df.columns:
        print("\nNach Kategorie:")
        print(
            df.groupby("math_category")["is_correct"]
            .mean()
            .apply(lambda x: f"{x:.1%}")
        )


if __name__ == "__main__":
    run_benchmark()
