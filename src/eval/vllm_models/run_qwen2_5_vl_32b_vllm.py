#!/usr/bin/env python3
"""
VLM Benchmark für Känguru-Mathematik-Aufgaben
Modell: Qwen2.5-VL-32B (vLLM Backend mit Structured Outputs / JSON Schema)

Verwendet Structured Outputs für garantierte JSON-Ausgabe.
Kompatibel mit vLLM >= 0.6.0 (nutzt guided_json Parameter).
"""

import os
import json
import logging
import re
import time
import random
import gc
import base64
import pandas as pd
from PIL import Image
from typing import Dict, List, Literal, Union
from pathlib import Path
from tqdm import tqdm
from enum import Enum

from pydantic import BaseModel, ValidationError, Field

# Projekt-Root ZUERST ermitteln (vor dotenv laden)
_script_path = Path(__file__).resolve()
PROJECT_ROOT = Path(os.environ.get("VLM_PROJECT_ROOT", _script_path.parent.parent.parent.parent))

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

# vLLM Import
from vllm import LLM, SamplingParams

# Versuche GuidedDecodingParams zu importieren (für neuere vLLM Versionen)
try:
    from vllm.sampling_params import GuidedDecodingParams
    VLLM_HAS_GUIDED_DECODING = True
    print("✅ GuidedDecodingParams verfügbar")
except ImportError:
    VLLM_HAS_GUIDED_DECODING = False
    print("ℹ️ GuidedDecodingParams nicht verfügbar - nutze guided_json direkt")

# ============================================================================
# KONFIGURATION - DIESES MODELL
# ============================================================================

MODEL_NAME = "Qwen2.5-VL-32B-vLLM"
MODEL_HF_ID = "Qwen/Qwen2.5-VL-32B-Instruct"
MODEL_PARAMS_B = 32

# Cache-Verzeichnis für Modelle (auf Cluster: /netscratch)
MODEL_CACHE_DIR = os.environ.get("HF_HOME", "/netscratch/$USER/.cache/huggingface")

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
SEED = 42

# ============================================================================
# PYDANTIC SCHEMA FÜR GUIDED DECODING
# ============================================================================

class AnswerChoice(str, Enum):
    """Enum für die möglichen Antworten."""
    A = "A"
    B = "B"
    C = "C"
    D = "D"
    E = "E"


class KanguruAnswer(BaseModel):
    """
    Antwort-Schema für Känguru-Mathematik Multiple-Choice.
    Wird für GuidedDecodingParams(json=...) verwendet.
    """
    answer: AnswerChoice = Field(
        description="Die korrekte Antwort: A, B, C, D oder E"
    )


# JSON Schema für guided decoding extrahieren
ANSWER_JSON_SCHEMA = KanguruAnswer.model_json_schema()

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


def free_gpu_memory():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except ImportError:
        pass
    time.sleep(1)


# ============================================================================
# ANTWORT-PARSING MIT PYDANTIC
# ============================================================================

def parse_response(output_text: str) -> Dict[str, Union[str, bool, None]]:
    """
    Extrahiert und validiert die Antwort aus dem JSON-Output.
    
    Mit guided_decoding(json=schema) ist die Ausgabe garantiert ein valides JSON
    im Format {"answer": "A/B/C/D/E"}.
    """
    clean_text = output_text.strip()
    
    # Direkter JSON-Parse (sollte durch guided decoding immer funktionieren)
    try:
        data = json.loads(clean_text)
        validated = KanguruAnswer(**data)
        return {"prediction": validated.answer.value, "format_valid": True, "error": None}
    except json.JSONDecodeError as e:
        logger.warning(f"JSON Parse Error: {e} - Text: {clean_text[:100]}")
    except ValidationError as e:
        error_msg = e.errors()[0].get('msg', str(e)) if e.errors() else str(e)
        logger.warning(f"Validation Error: {error_msg}")
        return {"prediction": None, "format_valid": False, "error": f"Schema violation: {error_msg}"}
    
    # Fallback: JSON-Objekt aus Text extrahieren
    json_match = re.search(r'\{[^{}]*\}', clean_text, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group(0))
            validated = KanguruAnswer(**data)
            return {"prediction": validated.answer.value, "format_valid": True, "error": None}
        except (json.JSONDecodeError, ValidationError):
            pass
    
    # Fallback: Direkter Buchstaben-Match
    for letter in ['A', 'B', 'C', 'D', 'E']:
        if letter in clean_text.upper():
            return {"prediction": letter, "format_valid": False, "error": "Extracted from text"}
    
    return {"prediction": None, "format_valid": False, "error": "No valid answer found"}


# ============================================================================
# BILD LADEN
# ============================================================================

def load_image_base64(image_path: Path) -> str:
    """Lädt ein Bild und gibt es als Base64-String zurück."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def get_image_mime_type(image_path: Path) -> str:
    """Ermittelt den MIME-Type basierend auf der Dateiendung."""
    suffix = image_path.suffix.lower()
    mime_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
    }
    return mime_types.get(suffix, "image/jpeg")


# ============================================================================
# EVALUATOR
# ============================================================================

class VLMEvaluator:
    def __init__(self):
        logger.info(f"🏗️ Lade {MODEL_NAME} ({MODEL_PARAMS_B}B) mit vLLM")
        logger.info(f"   HuggingFace ID: {MODEL_HF_ID}")
        logger.info(f"   ⚡ Guided Decoding (JSON Schema) aktiviert")

        # vLLM LLM initialisieren
        logger.info("   📥 Lade Modell mit vLLM...")
        self.llm = LLM(
            model=MODEL_HF_ID,
            trust_remote_code=True,
            max_model_len=4096,
            gpu_memory_utilization=0.9,
            dtype="bfloat16",
        )
        
        # Sampling Parameter erstellen - je nach vLLM Version
        if VLLM_HAS_GUIDED_DECODING:
            # Neuere vLLM Version: GuidedDecodingParams
            logger.info("   📋 Nutze GuidedDecodingParams (neuere vLLM)")
            guided_params = GuidedDecodingParams(json=ANSWER_JSON_SCHEMA)
            self.sampling_params = SamplingParams(
                max_tokens=50,
                temperature=0.0,
                guided_decoding=guided_params,
            )
        else:
            # Ältere vLLM Version: guided_json direkt in SamplingParams
            logger.info("   📋 Nutze guided_json direkt (ältere vLLM)")
            try:
                self.sampling_params = SamplingParams(
                    max_tokens=50,
                    temperature=0.0,
                    guided_json=ANSWER_JSON_SCHEMA,
                )
            except TypeError:
                # Falls guided_json auch nicht verfügbar ist
                logger.warning("   ⚠️ Keine Guided Decoding Unterstützung - nutze Fallback")
                self.sampling_params = SamplingParams(
                    max_tokens=50,
                    temperature=0.0,
                )
        
        logger.info(f"✅ {MODEL_NAME} bereit mit vLLM + JSON Schema Guided Decoding")
        logger.info(f"   Schema: {ANSWER_JSON_SCHEMA}")

    def generate(self, image_path: str) -> Dict:
        full_path = DATA_DIR / image_path
        if not full_path.exists():
            raise FileNotFoundError(f"Bild nicht gefunden: {full_path}")
        
        # Bild als Base64 laden
        image_b64 = load_image_base64(full_path)
        mime_type = get_image_mime_type(full_path)
        
        # Prompt mit Hinweis auf JSON-Format
        system_prompt = (
            "Du bist ein mathematisches Assistenzsystem für Multiple-Choice-Aufgaben.\n"
            "Analysiere das Bild und wähle die korrekte Antwort: A, B, C, D oder E.\n\n"
            "Antworte im JSON-Format: {\"answer\": \"X\"} wobei X = A, B, C, D oder E."
        )
        user_prompt = "Bestimme die richtige Antwort. Gib deine Antwort als JSON zurück."

        # OpenAI-kompatibles Message-Format für vLLM
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{image_b64}"
                        }
                    },
                    {
                        "type": "text",
                        "text": user_prompt
                    }
                ]
            }
        ]
        
        start_time = time.time()
        
        # vLLM Chat Completion mit Guided Decoding
        outputs = self.llm.chat(
            messages=messages,
            sampling_params=self.sampling_params,
        )
        
        duration = time.time() - start_time
        
        # Ausgabe extrahieren
        output_text = outputs[0].outputs[0].text.strip()
        input_tokens = len(outputs[0].prompt_token_ids) if outputs[0].prompt_token_ids else 0
        
        # JSON parsen und validieren
        result = parse_response(output_text)
        
        return {
            "prediction": result["prediction"],
            "format_valid": result["format_valid"],
            "error": result["error"],
            "inference_time": round(duration, 4),
            "input_tokens": input_tokens,
            "raw_output": output_text
        }

    def cleanup(self):
        logger.info(f"🧹 Räume {MODEL_NAME} auf...")
        if hasattr(self, 'llm'):
            del self.llm
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
                        "inference_time": result["inference_time"],
                        "input_tokens": result["input_tokens"],
                        "raw_output": result.get("raw_output", "")
                    }
                    
                    f_log.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
                    f_log.flush()
                    
                    acc = correct_count / processed_count if processed_count > 0 else 0
                    status = "✓" if is_correct else "✗"
                    pbar.set_postfix({"acc": f"{acc:.1%}", "last": f"{status} GT:{ground_truth} P:{result['prediction']}"})
                    
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
    print(f"📜 Logs:  {LOG_FILE}")


if __name__ == "__main__":
    run_benchmark()
    generate_report()
