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
    print("Starte Analyse...")
    if not RESULTS_DIR.exists():
        print(f"Ordner {RESULTS_DIR} nicht gefunden!")
        return

    import csv
    import csv
    model_data = {}
    for file_path in RESULTS_DIR.glob("*.jsonl"):
        print(f"Verarbeite {file_path.name}...")
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    model = entry.get('model')
                    if not model:
                        filename = file_path.stem
                        model = filename.replace('_results', '').replace('_', '-')
                    if not model:
                        continue
                    if model not in model_data:
                        model_data[model] = {
                            'model': model,
                            'total': 0,
                            'correct': 0,
                            'invalid_formats': 0,
                            'null_prediction': 0,
                            'valid': 0,
                            'categories': defaultdict(lambda: {'total': 0, 'correct': 0})
                        }
                    model_data[model]['total'] += 1
                    cat = entry.get('math_category', 'unknown')
                    if entry.get('prediction', 'NOT_SET') is None:
                        model_data[model]['null_prediction'] += 1
                        model_data[model]['categories'][cat]['total'] += 1
                        # null_prediction als Fehler, also correct bleibt 0
                        model_data[model]['categories'][cat]['null_prediction'] = model_data[model]['categories'][cat].get('null_prediction', 0) + 1
                        continue
                    model_data[model]['valid'] += 1
                    if entry.get('is_correct', False):
                        model_data[model]['correct'] += 1
                    if not entry.get('format_valid', True):
                        model_data[model]['invalid_formats'] += 1
                    # Kategorie-Statistik
                    model_data[model]['categories'][cat]['total'] += 1
                    if entry.get('is_correct', False):
                        model_data[model]['categories'][cat]['correct'] += 1
                except json.JSONDecodeError:
                    print(f"Fehler beim Parsen einer Zeile in {file_path.name}")

    if not model_data:
        print("Keine gültigen Einträge gefunden.")
        return

    # Schreibe aggregierte CSV direkt ins vlm_repo
    csv_path = Path(__file__).parent.parent / "auswertung_aggregiert.csv"
    with open(csv_path, 'w', encoding='utf-8', newline='') as csvfile:
        fieldnames = ['model', 'category', 'total', 'correct', 'accuracy', 'invalid_formats', 'null_prediction']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for model, data in sorted(model_data.items()):
            # Modell-Gesamtzeile
            accuracy = data['correct'] / data['total'] if data['total'] > 0 else 0
            writer.writerow({
                'model': model,
                'category': 'ALL',
                'total': data['total'],
                'correct': data['correct'],
                'accuracy': f"{accuracy:.4f}",
                'invalid_formats': data['invalid_formats'],
                'null_prediction': data['null_prediction']
            })
            # Pro Kategorie
            for cat, cat_data in sorted(data['categories'].items(), key=lambda x: str(x[0]) if x[0] is not None else ""):
                cat_total = cat_data['total']
                cat_null = cat_data.get('null_prediction', 0)
                # null_prediction als Fehler: total = alle, correct = nur korrekte
                cat_correct = cat_data['correct']
                cat_accuracy = cat_correct / cat_total if cat_total > 0 else 0
                writer.writerow({
                    'model': model,
                    'category': cat,
                    'total': cat_total,
                    'correct': cat_correct,
                    'accuracy': f"{cat_accuracy:.4f}",
                    'invalid_formats': '',
                    'null_prediction': cat_null
                })
    print(f"Aggregierte CSV gespeichert unter: {csv_path}")

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
            cat_null = cat_data.get('null_prediction', 0)
            cat_acc = cat_correct / cat_total if cat_total > 0 else 0
            print(f"      {cat}: {cat_correct}/{cat_total} = {cat_acc:.1%} (null_predictions: {cat_null})")

if __name__ == "__main__":
    analyze_results()