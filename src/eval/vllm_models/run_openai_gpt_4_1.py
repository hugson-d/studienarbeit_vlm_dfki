#!/usr/bin/env python3
"""
OpenAI Zero-Shot Benchmark für Känguru-Mathematik-Aufgaben.

Vergleichbar mit den bestehenden vLLM Zero-Shot Runs:
- Kein CoT
- Kein Voting
- Gleiches Prompting (Deutsch, System + User)
- Gleiches Antwortschema {"answer": "A|B|C|D|E"}
- Vergleichbares JSONL-Outputschema
"""

import argparse
import base64
import json
import logging
import os
import random
import time
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from openai import OpenAI
from pydantic import BaseModel, Field
from tqdm import tqdm

# Projekt-Root zuerst ermitteln (vor dotenv)
_script_path = Path(__file__).resolve()
PROJECT_ROOT = Path(
    os.environ.get("VLM_PROJECT_ROOT", _script_path.parent.parent.parent.parent)
)

# .env laden (optional)
try:
    from dotenv import load_dotenv

    _env_file = PROJECT_ROOT / ".env"
    if _env_file.exists():
        load_dotenv(_env_file, override=True)
    else:
        load_dotenv(override=True)
except ImportError:
    pass


DEFAULT_MODEL = "gpt-5.4"
DEFAULT_DATASET_PATH = PROJECT_ROOT / "dataset_final.json"
DEFAULT_OUTPUT_FILE = PROJECT_ROOT / "results_vllm" / "gpt-5.4_results.jsonl"
DATA_DIR = PROJECT_ROOT / "data"
SEED = 42

# Exakt wie in den Zero-Shot vLLM Skripten
SYSTEM_PROMPT = (
    "Du bist ein mathematisches Assistenzsystem für Multiple-Choice-Aufgaben.\n"
    "Analysiere das Bild und wähle die korrekte Antwort: A, B, C, D oder E.\n\n"
    "Antworte im JSON-Format: {\"answer\": \"X\"} wobei X = A, B, C, D oder E."
)
USER_PROMPT = "Bestimme die richtige Antwort. Gib deine Antwort als JSON zurück."
VALID_PREDICTIONS = {"A", "B", "C", "D", "E"}


class AnswerChoice(str, Enum):
    A = "A"
    B = "B"
    C = "C"
    D = "D"
    E = "E"


class KanguruAnswer(BaseModel):
    answer: AnswerChoice = Field(
        description="Die korrekte Antwort: A, B, C, D oder E"
    )


def set_seed(seed: int) -> None:
    random.seed(seed)


def get_image_mime_type(image_path: Path) -> str:
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


def load_image_base64(image_path: Path) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def create_task_id(item: Dict[str, Any]) -> str:
    year = item.get("year", "unknown")
    cls = item.get("class", "unknown")
    task_id = item.get("task_id", "unknown")
    return f"{year}_{cls}_{task_id}"


def has_valid_prediction(entry: Dict[str, Any]) -> bool:
    return entry.get("prediction") in VALID_PREDICTIONS


def load_existing_entries(output_file: Path) -> Dict[str, Dict[str, Any]]:
    entries_by_task: Dict[str, Dict[str, Any]] = {}
    if not output_file.exists():
        return entries_by_task

    with open(output_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            task_id = entry.get("task_id")
            if isinstance(task_id, str):
                # Letzte Zeile pro task_id gewinnt
                entries_by_task[task_id] = entry
    return entries_by_task


def write_entries_overwrite(
    output_file: Path,
    dataset: List[Dict[str, Any]],
    entries_by_task: Dict[str, Dict[str, Any]],
) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    tmp_file = output_file.with_suffix(output_file.suffix + ".tmp")

    written: Set[str] = set()
    with open(tmp_file, "w", encoding="utf-8") as f:
        for item in dataset:
            task_id = create_task_id(item)
            entry = entries_by_task.get(task_id)
            if entry is None:
                continue
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            written.add(task_id)

        # Fallback für alte/abweichende task_ids, die nicht im Dataset matchen
        for task_id, entry in entries_by_task.items():
            if task_id in written:
                continue
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    tmp_file.replace(output_file)


def load_dataset(dataset_path: Path) -> List[Dict[str, Any]]:
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset nicht gefunden: {dataset_path}")
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Ungültiges Dataset-Format in {dataset_path}: erwartete Liste.")
    return data


def build_logger(log_file: Path) -> logging.Logger:
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("OpenAI-ZeroShot")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


class OpenAIEvaluator:
    def __init__(self, model: str):
        self.model = model
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def generate(
        self,
        image_rel_path: str,
        reasoning_effort: Optional[str] = None,
    ) -> Dict[str, Any]:
        full_path = DATA_DIR / image_rel_path
        if not full_path.exists():
            return {
                "prediction": None,
                "format_valid": False,
                "error": f"Bild nicht gefunden: {full_path}",
                "inference_time": 0.0,
                "input_tokens": None,
                "raw_output": "",
            }

        image_b64 = load_image_base64(full_path)
        mime_type = get_image_mime_type(full_path)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{image_b64}",
                        },
                    },
                    {"type": "text", "text": USER_PROMPT},
                ],
            },
        ]

        start_time = time.time()
        attempts: List[Optional[str]] = [reasoning_effort]
        if reasoning_effort != "none":
            attempts.append("none")

        last_error = "Unbekannter Fehler"
        for effort in attempts:
            try:
                request_kwargs: Dict[str, Any] = {}
                if effort:
                    request_kwargs["reasoning_effort"] = effort
                # GPT-5.* currently accepts only default temperature.
                if not self.model.startswith("gpt-5"):
                    request_kwargs["temperature"] = 0.0

                completion = self.client.chat.completions.parse(
                    model=self.model,
                    messages=messages,
                    response_format=KanguruAnswer,
                    max_completion_tokens=512,
                    **request_kwargs,
                )
                duration = round(time.time() - start_time, 4)

                message = completion.choices[0].message
                parsed: Optional[KanguruAnswer] = getattr(message, "parsed", None)
                refusal: Optional[str] = getattr(message, "refusal", None)

                if parsed is not None:
                    raw_output = message.content or json.dumps(
                        parsed.model_dump(), ensure_ascii=False
                    )
                    return {
                        "prediction": parsed.answer.value,
                        "format_valid": True,
                        "error": None,
                        "inference_time": duration,
                        "input_tokens": getattr(completion.usage, "prompt_tokens", None),
                        "raw_output": raw_output,
                    }

                if refusal:
                    return {
                        "prediction": None,
                        "format_valid": False,
                        "error": f"Refusal: {refusal}",
                        "inference_time": duration,
                        "input_tokens": getattr(completion.usage, "prompt_tokens", None),
                        "raw_output": refusal,
                    }

                raw_output = message.content or ""
                return {
                    "prediction": None,
                    "format_valid": False,
                    "error": "Parse fehlgeschlagen: keine strukturierte Antwort erhalten",
                    "inference_time": duration,
                    "input_tokens": getattr(completion.usage, "prompt_tokens", None),
                    "raw_output": raw_output,
                }
            except Exception as e:
                err = str(e)
                last_error = err
                # Spezifischer Retry-Fix für den gemeldeten Fehler:
                # "Could not parse response content as the length limit was reached"
                if (
                    "length limit was reached" in err
                    and effort != "none"
                ):
                    continue
                break

        duration = round(time.time() - start_time, 4)
        return {
            "prediction": None,
            "format_valid": False,
            "error": last_error,
            "inference_time": duration,
            "input_tokens": None,
            "raw_output": last_error,
        }


def run_benchmark(
    model: str,
    dataset_path: Path,
    output_file: Path,
    max_tasks: Optional[int],
    reasoning_effort: Optional[str],
    logger: logging.Logger,
) -> None:
    set_seed(SEED)
    dataset = load_dataset(dataset_path)
    total_tasks = len(dataset)

    entries_by_task = load_existing_entries(output_file)
    valid_task_ids = {
        task_id for task_id, entry in entries_by_task.items() if has_valid_prediction(entry)
    }

    remaining_items = [item for item in dataset if create_task_id(item) not in valid_task_ids]
    if max_tasks is not None:
        remaining_items = remaining_items[:max_tasks]

    logger.info(f"📂 Dataset geladen: {total_tasks} Aufgaben")
    logger.info(f"📜 Output: {output_file}")
    logger.info(f"🤖 Modell: {model}")
    logger.info(f"✅ Bereits valide Predictions: {len(valid_task_ids)}")
    logger.info(f"🚀 Offene Tasks: {len(remaining_items)}")

    if not remaining_items:
        logger.info("⏭️ Keine offenen Tasks. Fertig.")
        return

    evaluator = OpenAIEvaluator(model=model)
    correct_count = 0
    processed_count = 0

    pbar = tqdm(remaining_items, desc=model, unit="task")
    stop_due_quota = False
    for idx, item in enumerate(pbar, start=1):
        task_id = create_task_id(item)
        existing_entry = entries_by_task.get(task_id)
        if existing_entry is not None and has_valid_prediction(existing_entry):
            # Harte Schutzregel: bereits valide Predictions niemals überschreiben
            continue

        result = evaluator.generate(
            item.get("image_path", ""),
            reasoning_effort=reasoning_effort,
        )

        ground_truth = item.get("answer")
        prediction = result.get("prediction")
        is_correct = prediction is not None and prediction == ground_truth

        if is_correct:
            correct_count += 1
        processed_count += 1

        log_entry = {
            "model": model,
            "task_id": task_id,
            "year": item.get("year"),
            "class": item.get("class"),
            "original_task_id": item.get("task_id"),
            "math_category": item.get("math_category"),
            "is_text_only": item.get("is_text_only"),
            "ground_truth": ground_truth,
            "prediction": prediction,
            "is_correct": is_correct,
            "format_valid": result.get("format_valid"),
            "error_type": result.get("error"),
            "inference_time": result.get("inference_time"),
            "input_tokens": result.get("input_tokens"),
            "raw_output": result.get("raw_output", ""),
        }

        # Nur dann überschreiben, wenn wir etwas Besseres haben:
        # - es gibt noch keinen Eintrag, oder
        # - die neue Prediction ist valide (A-E)
        if task_id not in entries_by_task:
            entries_by_task[task_id] = log_entry
        elif not has_valid_prediction(entries_by_task[task_id]) and has_valid_prediction(log_entry):
            entries_by_task[task_id] = log_entry

        # Periodisch auf Disk schreiben (sicher gegen Abbruch)
        if idx % 20 == 0:
            write_entries_overwrite(output_file, dataset, entries_by_task)

        acc = correct_count / processed_count if processed_count else 0.0
        status = "✓" if is_correct else "✗"
        pbar.set_postfix(
            {"acc": f"{acc:.1%}", "last": f"{status} GT:{ground_truth} P:{prediction}"}
        )

        error_text = str(result.get("error") or "")
        if "insufficient_quota" in error_text:
            logger.error("💸 API quota exhausted (insufficient_quota). Stoppe Lauf.")
            stop_due_quota = True
            break
    pbar.close()

    # Final write
    write_entries_overwrite(output_file, dataset, entries_by_task)

    if stop_due_quota:
        logger.info("⏹️ Lauf wegen quota exhaustion beendet.")

    logger.info(
        f"📊 Ergebnis (neu verarbeitet): {correct_count}/{processed_count} = "
        f"{(correct_count / processed_count if processed_count else 0):.1%}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="OpenAI Zero-Shot Runner für Känguru-Mathematik Benchmark."
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"OpenAI Modellname (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help=f"Pfad zu dataset_final.json (default: {DEFAULT_DATASET_PATH})",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=DEFAULT_OUTPUT_FILE,
        help=f"Pfad zur Ergebnis-JSONL (default: {DEFAULT_OUTPUT_FILE})",
    )
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=None,
        help="Optionales Limit für neu zu verarbeitende Tasks (Smoke Test).",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=["none", "low", "medium", "high", "xhigh"],
        default="low",
        help="Reasoning effort for supported models (default: low).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    log_file = args.output_file.with_suffix(".log")
    logger = build_logger(log_file)

    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY nicht gesetzt.")

    run_benchmark(
        model=args.model,
        dataset_path=args.dataset_path,
        output_file=args.output_file,
        max_tasks=args.max_tasks,
        reasoning_effort=args.reasoning_effort,
        logger=logger,
    )


if __name__ == "__main__":
    main()
