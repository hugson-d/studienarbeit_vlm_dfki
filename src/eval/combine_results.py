#!/usr/bin/env python3
"""
Kombiniert alle Einzel-Ergebnisse zu einem Gesamt-Report.
Führe dieses Skript aus, nachdem alle Modelle durchgelaufen sind.
"""

import json
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "evaluation_results"

# Alle Modelle
MODELS = [
    "Qwen2.5-VL-72B", "Qwen2.5-VL-32B", "Qwen2.5-VL-7B", "Qwen2.5-VL-3B",
    "InternVL3-78B", "InternVL3-38B", "InternVL3-14B", "InternVL3-8B",
    "Ovis2.5-9B", "Ovis2.5-2B",
    "Ovis2-34B", "Ovis2-16B", "Ovis2-8B", "Ovis2-4B"
]


def combine_results():
    """Kombiniert alle Ergebnisse zu einer Datei."""
    all_data = []
    
    print("📂 Lade Ergebnisse...")
    for model in MODELS:
        log_file = OUTPUT_DIR / f"{model}_results.jsonl"
        if log_file.exists():
            with open(log_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        all_data.append(entry)
                    except:
                        pass
            print(f"  ✅ {model}: {sum(1 for e in all_data if e.get('model') == model)} Einträge")
        else:
            print(f"  ⚠️ {model}: Keine Ergebnisse gefunden")
    
    if not all_data:
        print("❌ Keine Daten gefunden!")
        return
    
    # DataFrame erstellen
    df = pd.DataFrame(all_data)
    
    # Gesamt-JSONL speichern
    combined_jsonl = OUTPUT_DIR / "all_results_combined.jsonl"
    with open(combined_jsonl, 'w', encoding='utf-8') as f:
        for entry in all_data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    
    # Excel speichern
    combined_excel = OUTPUT_DIR / "all_results_combined.xlsx"
    df.to_excel(combined_excel, index=False)
    
    print(f"\n📁 Kombinierte Dateien:")
    print(f"  JSONL: {combined_jsonl}")
    print(f"  Excel: {combined_excel}")
    
    return df


def generate_summary(df: pd.DataFrame):
    """Erstellt Summary-Statistiken."""
    print("\n" + "="*80)
    print("📊 GESAMT-ERGEBNISSE ALLER MODELLE")
    print("="*80)
    
    # Hauptmetriken
    summary = df.groupby('model').agg({
        'is_correct': ['mean', 'sum', 'count'],
        'format_valid': 'mean',
        'inference_time': 'mean'
    }).round(4)
    summary.columns = ['Accuracy', 'Correct', 'Total', 'Valid JSON %', 'Avg Time (s)']
    summary = summary.sort_values('Accuracy', ascending=False)
    
    print("\n📈 Ranking nach Accuracy:")
    print(summary.to_string())
    
    # Nach Modell-Familie
    print("\n📊 Nach Modell-Familie:")
    df['family'] = df['model'].apply(lambda x: x.split('-')[0] if '-' in x else x)
    family_summary = df.groupby('family')['is_correct'].mean().sort_values(ascending=False)
    for family, acc in family_summary.items():
        print(f"  {family:20s} {acc:.1%}")
    
    # Nach Kategorie
    if 'math_category' in df.columns:
        print("\n📐 Accuracy nach Mathematischer Kategorie:")
        cat_summary = df.groupby(['model', 'math_category'])['is_correct'].mean().unstack().round(3)
        print(cat_summary.to_string())
    
    # Nach Klassenstufe
    if 'class' in df.columns:
        print("\n🎓 Accuracy nach Klassenstufe:")
        class_summary = df.groupby(['model', 'class'])['is_correct'].mean().unstack().round(3)
        print(class_summary.to_string())
    
    # Text-Only vs Visual
    if 'is_text_only' in df.columns:
        print("\n👁️ Text-Only vs. Visuell:")
        vis_summary = df.groupby(['model', 'is_text_only'])['is_correct'].mean().unstack().round(3)
        vis_summary.columns = ['Visuell', 'Text-Only']
        print(vis_summary.to_string())
    
    # Summary Excel speichern
    summary_excel = OUTPUT_DIR / "benchmark_summary.xlsx"
    with pd.ExcelWriter(summary_excel) as writer:
        summary.to_excel(writer, sheet_name='Übersicht')
        if 'math_category' in df.columns:
            cat_summary.to_excel(writer, sheet_name='Nach Kategorie')
        if 'class' in df.columns:
            class_summary.to_excel(writer, sheet_name='Nach Klassenstufe')
    
    print("\n" + "="*80)
    print(f"📁 Summary Excel: {summary_excel}")
    print("="*80)


def main():
    df = combine_results()
    if df is not None and not df.empty:
        generate_summary(df)


if __name__ == "__main__":
    main()
