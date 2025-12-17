#!/usr/bin/env python3
"""
Patcht die Datei results_vllm/mistral-small-2506_structured_api_results.jsonl direkt (In-Place-Update).
Gleicht task_id mit dem Dateinamen aus dataset_final.json ab und
erzwingt das Setzen von 'is_text_only' und 'math_category'.
"""
import json
from pathlib import Path

def build_mapping_from_jsonl(jsonl_path):
    """
    Erstellt Mapping: task_id -> {is_text_only, math_category}
    Beispiel Key: "1998_3und4_A1" (aus Gemma-3-27B-vLLM_results.jsonl)
    """
    print(f"Lade Mapping aus {jsonl_path} ...")
    mapping = {}
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)
            task_id = entry.get("task_id")
            if task_id:
                mapping[task_id] = {
                    "math_category": entry.get("math_category"),
                    "is_text_only": entry.get("is_text_only")
                }
    return mapping

def patch_file_inplace(jsonl_path, mapping):
    jsonl_path = Path(jsonl_path)

    # 1. Alles in den Speicher lesen
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    patched_lines = []
    changes_count = 0

    # 2. Daten verarbeiten
    for line in lines:
        if not line.strip():
            continue

        entry = json.loads(line)
        task_id = entry.get("task_id") # z.B. "2025_11bis13_B7"

        if task_id:
            if task_id in mapping:
                source = mapping[task_id]

                # Werte hart überschreiben (oder neu anlegen)
                entry["is_text_only"] = source["is_text_only"]
                entry["math_category"] = source["math_category"]
                changes_count += 1
            else:
                print(f"Warnung: task_id {task_id} nicht im Mapping gefunden.")

        patched_lines.append(json.dumps(entry, ensure_ascii=False))

    # 3. Datei direkt überschreiben
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(patched_lines) + "\n")

    print(f"Updated: {jsonl_path.name} (Einträge aktualisiert: {changes_count})")

def main():
    mapping_file = Path("results_vllm/Gemma-3-27B-vLLM_results.jsonl")
    if not mapping_file.exists():
        print("Fehler: Mapping-Datei fehlt.")
        return

    mapping = build_mapping_from_jsonl(mapping_file)

    # Process only the specified file
    jsonl_file = "results_vllm/mistral-small-2506_structured_api_results.jsonl"
    if not Path(jsonl_file).exists():
        print(f"Fehler: {jsonl_file} fehlt.")
        return

    print(f"Verarbeite Datei: {jsonl_file}...")
    patch_file_inplace(jsonl_file, mapping)

    print("Fertig. Datei wurde überschrieben.")

if __name__ == "__main__":
    main()