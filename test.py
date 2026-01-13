#!/usr/bin/env python3
"""
Auswertung: Warum prediction == null?
Liest JSONL im selben Pfad und gibt Häufigkeiten + Beispiele aus.

Datei (fix):
  InternVL3-14B_FailureAnalysis_1run_20260112_195910_results.jsonl

Run:
  python3 analyze_null_predictions.py
"""

import json
import re
from pathlib import Path
from collections import Counter, defaultdict

FILE = Path("InternVL3-14B_FailureAnalysis_1run_20260112_195910_results.jsonl")

def safe_get(d, *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur

def msg_prefix(s: str, n: int = 80) -> str:
    s = (s or "").strip().replace("\n", "\\n")
    return s[:n]

def classify_root_cause(rec: dict) -> str:
    """
    Grobe Klassifikation für die häufigsten Ursachen.
    Kombiniert:
      - failure_stage
      - finish_reason
      - typische json error messages
      - pydantic validation patterns
    """
    stage = (safe_get(rec, "raw_output_analysis", "failure_stage") or "MISSING").strip()
    finish = (rec.get("finish_reason") or "MISSING").strip()
    msg = (safe_get(rec, "raw_output_analysis", "failure_message") or "").strip()

    # Dataset / file issues
    if rec.get("error_category") == "file_not_found":
        return "file_not_found"
    if stage == "dataset":
        return "dataset_error"

    # Empty outputs
    if stage == "empty_output":
        return "empty_output"

    # JSON parse problems
    if stage == "json_parse":
        # häufigster Fall in deinem Beispiel: Output wurde wegen max_tokens abgeschnitten
        if finish == "length":
            if re.search(r"Unterminated string", msg, re.IGNORECASE):
                return "json_truncated_unterminated_string (finish=length)"
            if re.search(r"Expecting ',' delimiter", msg, re.IGNORECASE):
                return "json_truncated_missing_comma_or_cutoff (finish=length)"
            if re.search(r"Expecting value", msg, re.IGNORECASE):
                return "json_truncated_expecting_value (finish=length)"
            return "json_parse_failed_finish_length"

        # nicht-length: anderes Problem (z.B. Modell druckt Text vor/nach JSON)
        if re.search(r"Extra data", msg, re.IGNORECASE):
            return "json_extra_data_before_or_after_json"
        if re.search(r"Expecting value", msg, re.IGNORECASE):
            return "json_expecting_value"
        if re.search(r"Unterminated string", msg, re.IGNORECASE):
            return "json_unterminated_string"
        if re.search(r"Expecting property name enclosed in double quotes", msg, re.IGNORECASE):
            return "json_bad_quotes_or_trailing_comma"
        if re.search(r"Expecting ',' delimiter", msg, re.IGNORECASE):
            return "json_missing_comma"
        return "json_parse_other"

    # Schema validation problems (JSON ok, aber passt nicht zum Schema)
    if stage == "pydantic_validation":
        if re.search(r"Field required|missing", msg, re.IGNORECASE):
            return "schema_missing_required_field"
        if re.search(r"answer", msg, re.IGNORECASE) and re.search(r"Input should be|valid enumeration", msg, re.IGNORECASE):
            return "schema_answer_not_in_A_to_E"
        if re.search(r"type", msg, re.IGNORECASE):
            return "schema_wrong_type"
        return "schema_validation_other"

    # Unknown / fallback
    cat = (rec.get("error_category") or "unknown").strip()
    return f"other:{cat}:{stage}:{finish}"

def load_jsonl(path: Path):
    records = []
    bad_lines = 0
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except Exception:
                bad_lines += 1
    return records, bad_lines

def print_top(counter: Counter, title: str, k: int = 20):
    print(f"\n=== {title} (top {k}) ===")
    for key, n in counter.most_common(k):
        print(f"{n:6d}  {key}")

def main():
    if not FILE.exists():
        raise SystemExit(f"Datei nicht gefunden: {FILE.resolve()}")

    records, bad_lines = load_jsonl(FILE)
    total = len(records)

    null_recs = [r for r in records if r.get("prediction") is None]
    n_null = len(null_recs)

    print(f"File: {FILE.name}")
    print(f"Total records: {total}")
    print(f"Lines that were not valid JSON: {bad_lines}")
    print(f"prediction == null: {n_null}")

    if n_null == 0:
        print("Keine null-Predictions gefunden.")
        return

    # Häufigkeiten: error_category / failure_stage / finish_reason / failure_message
    c_error_category = Counter((r.get("error_category") or "MISSING") for r in null_recs)
    c_failure_stage = Counter((safe_get(r, "raw_output_analysis", "failure_stage") or "MISSING") for r in null_recs)
    c_finish_reason = Counter((r.get("finish_reason") or "MISSING") for r in null_recs)
    c_failure_message = Counter(msg_prefix(safe_get(r, "raw_output_analysis", "failure_message") or "MISSING", 120) for r in null_recs)

    # Root-cause Klassifikation
    root_labels = [classify_root_cause(r) for r in null_recs]
    c_root = Counter(root_labels)

    print_top(c_error_category, "error_category (prediction==null)")
    print_top(c_failure_stage, "raw_output_analysis.failure_stage (prediction==null)")
    print_top(c_finish_reason, "finish_reason (prediction==null)")
    print_top(c_root, "ROOT CAUSE (heuristisch klassifiziert)")
    print_top(c_failure_message, "failure_message (prefix, prediction==null)", k=15)

    # Beispiele für Top Root Cause
    top_root, top_root_n = c_root.most_common(1)[0]
    print(f"\n=== Beispiele für häufigste ROOT CAUSE: {top_root} (n={top_root_n}) ===")

    shown = 0
    for r in null_recs:
        if classify_root_cause(r) != top_root:
            continue
        task_id = r.get("task_id")
        finish = r.get("finish_reason")
        stage = safe_get(r, "raw_output_analysis", "failure_stage")
        msg = safe_get(r, "raw_output_analysis", "failure_message")
        raw = (r.get("raw_output") or "")
        raw_trunc = raw[:700].replace("\n", "\\n")

        print("\n----")
        print(f"task_id: {task_id}")
        print(f"finish_reason: {finish}")
        print(f"failure_stage: {stage}")
        print(f"failure_message: {msg}")
        print(f"raw_output_trunc(700): {raw_trunc}")

        shown += 1
        if shown >= 5:
            break

    # Optional: Summary als JSON speichern
    summary = {
        "file": FILE.name,
        "total_records": total,
        "bad_jsonl_lines": bad_lines,
        "null_predictions": n_null,
        "top_error_category": c_error_category.most_common(20),
        "top_failure_stage": c_failure_stage.most_common(20),
        "top_finish_reason": c_finish_reason.most_common(20),
        "top_root_causes": c_root.most_common(30),
        "top_failure_message_prefix": c_failure_message.most_common(20),
    }
    out_path = FILE.with_name(FILE.stem.replace("_results", "") + "_nullpred_summary.json")
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSummary geschrieben: {out_path.name}")

if __name__ == "__main__":
    main()
