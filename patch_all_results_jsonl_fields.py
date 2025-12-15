#!/usr/bin/env python3
"""
Patcht alle results_vllm/*.jsonl Dateien direkt (In-Place-Update).
Gleicht task_id mit dem Dateinamen aus dataset_final.json ab und
erzwingt das Setzen von 'is_text_only' und 'math_category'.
"""
import json
from pathlib import Path
import glob

def build_mapping(dataset_path):
    """
    Erstellt Mapping: Dateiname ohne Endung -> {is_text_only, math_category}
    Beispiel Key: "2025_11bis13_B7" (aus dataset_final/2025_11bis13_B7.png)
    """
    print(f"Lade Mapping aus {dataset_path} ...")
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    mapping = {}
    for item in data:
        image_path = item.get('image_path', '')
        if image_path:
            # Holt "2025_11bis13_B7" aus "dataset_final/2025_11bis13_B7.png"
            key = Path(image_path).stem 
            mapping[key] = {
                "math_category": item.get("math_category"),
                "is_text_only": item.get("is_text_only")
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
        
        if task_id and task_id in mapping:
            source = mapping[task_id]
            
            # Werte hart überschreiben (oder neu anlegen)
            entry["is_text_only"] = source["is_text_only"]
            entry["math_category"] = source["math_category"]
            changes_count += 1
            
        patched_lines.append(json.dumps(entry, ensure_ascii=False))
    
    # 3. Datei direkt überschreiben
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(patched_lines) + "\n")
        
    print(f"Updated: {jsonl_path.name} (Einträge aktualisiert: {changes_count})")

def main():
    dataset_file = Path("dataset_final.json")
    if not dataset_file.exists():
        print("Fehler: dataset_final.json fehlt.")
        return

    mapping = build_mapping(dataset_file)
    
    # Suche alle .jsonl Dateien im Unterordner
    jsonl_files = glob.glob("results_vllm/*.jsonl")
    
    if not jsonl_files:
        print("Keine .jsonl Dateien in results_vllm/ gefunden.")
        return

    print(f"Verarbeite {len(jsonl_files)} Dateien...")
    for f in jsonl_files:
        patch_file_inplace(f, mapping)
        
    print("Fertig. Alle Dateien wurden überschrieben.")

if __name__ == "__main__":
    main()