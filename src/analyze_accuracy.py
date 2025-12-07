#!/usr/bin/env python3
"""
Skript zur Berechnung der Accuracy und Format-Validität für alle JSONL-Dateien in results_vllm.
"""

import json
import os
from collections import defaultdict
from pathlib import Path

# Pfad zum results_vllm Ordner
RESULTS_DIR = Path("/Users/dennishug/Desktop/vlm_repo/results_vllm")

def analyze_results():
    if not RESULTS_DIR.exists():
        print(f"Ordner {RESULTS_DIR} nicht gefunden!")
        return

    # Sammle Daten pro Modell
    model_data = defaultdict(lambda: {
        'total': 0,
        'correct': 0,
        'categories': defaultdict(lambda: {'total': 0, 'correct': 0}),
        'invalid_formats': 0
    })

    # Gehe durch alle .jsonl Dateien
    for file_path in RESULTS_DIR.glob("*.jsonl"):
        print(f"Verarbeite {file_path.name}...")
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    model = entry.get('model')
                    if not model:
                        continue

                    data = model_data[model]
                    data['total'] += 1
                    if entry.get('is_correct', False):
                        data['correct'] += 1

                    # Pro Kategorie
                    category = entry.get('math_category', 'Unknown')
                    data['categories'][category]['total'] += 1
                    if entry.get('is_correct', False):
                        data['categories'][category]['correct'] += 1

                    # Format valid
                    if not entry.get('format_valid', True):
                        data['invalid_formats'] += 1

                except json.JSONDecodeError:
                    print(f"Fehler beim Parsen einer Zeile in {file_path.name}")

    # Ausgabe der Ergebnisse
    print("\n" + "="*80)
    print("ERGEBNISSE: Accuracy und Format-Validität pro Modell")
    print("="*80)

    for model, data in sorted(model_data.items()):
        total = data['total']
        correct = data['correct']
        accuracy = correct / total if total > 0 else 0
        invalid_formats = data['invalid_formats']

        print(f"\n📊 Modell: {model}")
        print(f"   Gesamt: {correct}/{total} = {accuracy:.1%}")
        print(f"   Ungültige Formate: {invalid_formats}")

        print("   📐 Pro math_category:")
        for cat, cat_data in sorted(data['categories'].items(), key=lambda x: str(x[0]) if x[0] is not None else ""):
            cat_total = cat_data['total']
            cat_correct = cat_data['correct']
            cat_acc = cat_correct / cat_total if cat_total > 0 else 0
            print(f"      {cat}: {cat_correct}/{cat_total} = {cat_acc:.1%}")

if __name__ == "__main__":
    analyze_results()