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
        # Robustes Bauen des Keys aus image_path + task_id
        # Beispiel: image_path = 'dataset_final/1998_3und4_1.png', task_id = 'A1'
        # -> key = '1998_3und4_A1'
        image_path = item.get('image_path', '')
        tid = item.get('task_id')
        key = None
        if image_path and tid:
            try:
                fname = Path(image_path).stem  # '1998_3und4_1'
                parts = fname.split('_')
                if len(parts) >= 2:
                    year = parts[0]
                    cls = parts[1]
                    key = f"{year}_{cls}_{tid}"
            except Exception:
                key = None
        # Fallback: falls key noch nicht gesetzt, versuchen mit year/class/task_id Feldern
        if key is None:
            year = item.get('year', 'unknown')
            cls = item.get('class', 'unknown')
            tid = tid or item.get('task_id', 'unknown')
            key = f"{year}_{cls}_{tid}"
        mapping[key] = {
            "math_category": item.get("math_category"),
            "is_text_only": item.get("is_text_only")
        }
    return mapping

# --- Patch-Funktion wie gehabt ---
def patch_jsonl(jsonl_path, mapping):
    jsonl_path = Path(jsonl_path)
    out_path = jsonl_path.with_suffix(jsonl_path.suffix + '.patched')
    unmatched = 0
    total = 0
    with open(jsonl_path, 'r', encoding='utf-8') as fin, open(out_path, 'w', encoding='utf-8') as fout:
        for line in fin:
            if not line.strip():
                continue
            total += 1
            entry = json.loads(line)
            task_id = entry.get("task_id")
            if task_id and (task_id in mapping):
                m = mapping[task_id]
                if ("math_category" not in entry) or (entry.get("math_category") is None):
                    entry["math_category"] = m.get("math_category")
                if ("is_text_only" not in entry) or (entry.get("is_text_only") is None):
                    entry["is_text_only"] = m.get("is_text_only")
            else:
                unmatched += 1
            fout.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"Wrote patched file: {out_path} (total={total}, unmatched={unmatched})")
    return out_path

# --- Main: Alle JSONL patchen ---
def main():
    dataset_path = Path("dataset_final.json")
    mapping = build_taskid_map(dataset_path)
    jsonl_files = glob.glob("results_vllm/*.jsonl")
    for jsonl_path in jsonl_files:
        patched = patch_jsonl(jsonl_path, mapping)
        # Optional: overwrite original (disabled by default). If desired, uncomment.
        # backup = Path(jsonl_path + '.bak')
        # Path(jsonl_path).replace(backup)
        # patched.replace(jsonl_path)

if __name__ == "__main__":
    main()
