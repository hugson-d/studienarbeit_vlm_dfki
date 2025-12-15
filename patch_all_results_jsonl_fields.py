#!/usr/bin/env python3
"""
Patch all results_vllm/*.jsonl files to ensure math_category and is_text_only fields by joining with dataset_final.json.
"""
import json
from pathlib import Path
import glob

# --- Mapping-Funktion wie gehabt ---
def build_taskid_map(dataset_path):
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    mapping = {}
    for item in data:
        year = item.get('year', 'unknown')
        cls = item.get('class', 'unknown')
        tid = item.get('task_id', 'unknown')
        key = f"{year}_{cls}_{tid}"
        mapping[key] = {
            "math_category": item.get("math_category"),
            "is_text_only": item.get("is_text_only")
        }
    return mapping

# --- Patch-Funktion wie gehabt ---
def patch_jsonl(jsonl_path, mapping):
    jsonl_path = Path(jsonl_path)
    with open(jsonl_path, 'r', encoding='utf-8') as fin:
        lines = fin.readlines()
    new_lines = []
    for line in lines:
        entry = json.loads(line)
        task_id = entry.get("task_id")
        if task_id and (task_id in mapping):
            if "math_category" not in entry or entry["math_category"] is None:
                entry["math_category"] = mapping[task_id]["math_category"]
            if "is_text_only" not in entry or entry["is_text_only"] is None:
                entry["is_text_only"] = mapping[task_id]["is_text_only"]
        new_lines.append(json.dumps(entry, ensure_ascii=False) + "\n")
    with open(jsonl_path, 'w', encoding='utf-8') as fout:
        fout.writelines(new_lines)
    print(f"Patched file overwritten: {jsonl_path}")

# --- Main: Alle JSONL patchen ---
def main():
    dataset_path = Path("dataset_final.json")
    mapping = build_taskid_map(dataset_path)
    jsonl_files = glob.glob("results_vllm/*.jsonl")
    for jsonl_path in jsonl_files:
        patch_jsonl(jsonl_path, mapping)

if __name__ == "__main__":
    main()
