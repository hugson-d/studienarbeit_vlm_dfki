import glob
import json
from collections import defaultdict

# Alle JSONL-Dateien im results_vllm-Ordner
jsonl_files = glob.glob("results_vllm/*.jsonl")

def find_task_id_duplicates(file_path):
    seen = set()
    duplicates = set()
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                task_id = obj.get("task_id")
                if task_id is not None:
                    if task_id in seen:
                        duplicates.add(task_id)
                    else:
                        seen.add(task_id)
            except Exception as e:
                print(f"Fehler in {file_path}: {e}")
    return duplicates

def main():
    any_duplicates = False
    for file in jsonl_files:
        dups = find_task_id_duplicates(file)
        if dups:
            any_duplicates = True
            print(f"Duplikate in {file}:")
            for tid in sorted(dups):
                print(f"  {tid}")
    if not any_duplicates:
        print("Keine Duplikate gefunden.")

if __name__ == "__main__":
    main()
