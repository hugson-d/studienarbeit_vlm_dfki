#!/usr/bin/env python3
import json
from pathlib import Path
from collections import Counter

def format_percent(count, total):
    if total == 0:
        return "0,0"
    val = (count / total) * 100
    return f"{val:.1f}".replace('.', ',')

def main():
    dataset_path = Path("dataset_final.json")
    
    if not dataset_path.exists():
        print(f"Fehler: {dataset_path} nicht gefunden.")
        return

    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Speicher für Statistiken
    diff_counts = Counter()  # Für A, B, C
    ans_counts = Counter()   # Für A, B, C, D, E
    
    visual_entries = []
    
    print("Analysiere Daten...")

    for entry in data:
        # 1. Filter: Nur Visueller Teildatensatz
        # Logik: Wenn is_text_only True ist -> Text. Alles andere -> Visuell.
        is_text_only = entry.get("is_text_only")
        if is_text_only is True:
            continue # Überspringen, da Text-Aufgabe
        
        visual_entries.append(entry)

        # 2. Schwierigkeitsgrad ermitteln
        # Annahme: task_id ist z.B. "A5", "B7", "C1" -> erster Buchstabe ist Level
        task_id = entry.get("task_id", "")
        if task_id:
            level = task_id[0].upper() # Nimm ersten Buchstaben
            if level in ['A', 'B', 'C']:
                diff_counts[level] += 1
            # Falls es task_ids gibt, die anders aussehen, werden sie ignoriert
        
        # 3. Antwortverteilung
        answer = entry.get("answer", "").upper()
        if answer in ['A', 'B', 'C', 'D', 'E']:
            ans_counts[answer] += 1

    N = len(visual_entries)
    print(f"\n--- ERGEBNISSE FÜR VISUELLEN TEILDATENSATZ (N={N}) ---\n")

    # --- Ausgabe für Tabelle Linke Seite (Schwierigkeit) ---
    print("Kopierbare Werte für 'Schwierigkeitsgrad':")
    for level in ['A', 'B', 'C']:
        cnt = diff_counts[level]
        perc = format_percent(cnt, N)
        # Mapping für Label
        label_map = {'A': 'A (Leicht)', 'B': 'B (Mittel)', 'C': 'C (Schwer)'}
        print(f"{label_map[level]} & {cnt} & {perc}\\,\\% \\\\")
    
    print("-" * 30)

    # --- Ausgabe für Tabelle Rechte Seite (Antworten) ---
    print("Kopierbare Werte für 'Antwortverteilung':")
    for opt in ['A', 'B', 'C', 'D', 'E']:
        cnt = ans_counts[opt]
        perc = format_percent(cnt, N)
        print(f"{opt} & {cnt} & {perc}\\,\\% \\\\")

    print("-" * 30)
    print(f"Gesamt N: {N}")

if __name__ == "__main__":
    main()