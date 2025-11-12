"""
Analyse-Script für extrahierte Aufgaben 1998-2011.
Vergleicht extrahierte Aufgaben mit erwarteten Lösungen und exportiert nach Excel.
"""

import json
from pathlib import Path
from collections import defaultdict
import pandas as pd


def load_solutions_1998_2011():
    """Lädt Lösungen für 1998-2011 (sequentielle Nummerierung 1-30)."""
    with open('lösungen_1998_2011.json') as f:
        solutions = json.load(f)
    return solutions


def load_extracted_tasks():
    """Lädt extrahierte Aufgaben aus JSONL."""
    manifest_path = Path('data/references_1998_2011/tasks_manifest_1998_2011.jsonl')
    
    if not manifest_path.exists():
        print(f"⚠ Manifest nicht gefunden: {manifest_path}")
        return []
    
    tasks = []
    with open(manifest_path) as f:
        for line in f:
            tasks.append(json.loads(line))
    
    return tasks


def compare_extracted_vs_expected():
    """Vergleicht extrahierte Tasks mit erwarteten Lösungen."""
    solutions = load_solutions_1998_2011()
    extracted = load_extracted_tasks()
    
    # Gruppiere solutions nach (Jahr, Klasse)
    expected_by_year_class = defaultdict(set)
    for sol in solutions:
        key = (sol['Jahr'], sol['Klasse'])
        expected_by_year_class[key].add(sol['Aufgabe'])
    
    # Gruppiere extracted nach (Jahr, Klasse)
    extracted_by_year_class = defaultdict(set)
    for task in extracted:
        key = (task['year'], task['class'])
        extracted_by_year_class[key].add(task['task_id'])
    
    # Expected counts pro Klasse (1998-2011)
    expected_counts = {
        '3 und 4': 21,
        '5 und 6': 30,
        '7 und 8': 30,
        '9 und 10': 30,
        '11 bis 13': 30
    }
    
    print("=" * 80)
    print("ANALYSE: Extrahierte Aufgaben 1998-2011")
    print("=" * 80)
    print()
    
    # Statistik nach Jahr
    print("📊 ÜBERSICHT NACH JAHR")
    print("-" * 80)
    
    years = sorted(set(sol['Jahr'] for sol in solutions))
    
    results_data = []
    
    for year in years:
        year_expected = sum(
            len(expected_by_year_class.get((year, cls), set()))
            for cls in expected_counts.keys()
        )
        year_extracted = sum(
            len(extracted_by_year_class.get((year, cls), set()))
            for cls in expected_counts.keys()
        )
        year_missing = year_expected - year_extracted
        
        status = "✓ OK" if year_missing == 0 else f"⚠ {year_missing} fehlen"
        
        print(f"{year}: {year_extracted:3d}/{year_expected:3d} extrahiert  {status}")
        
        results_data.append({
            'Jahr': year,
            'Extrahiert': year_extracted,
            'Erwartet': year_expected,
            'Fehlend': year_missing,
            'Prozent': f"{100*year_extracted/year_expected:.1f}%" if year_expected > 0 else "0%"
        })
    
    print()
    
    # Detaillierte Analyse nach Jahr und Klasse
    print("📊 DETAILLIERTE ANALYSE NACH JAHR UND KLASSE")
    print("-" * 80)
    
    detail_data = []
    
    for year in years:
        print(f"\nJahr {year}:")
        for class_name in ['3 und 4', '5 und 6', '7 und 8', '9 und 10', '11 bis 13']:
            expected_count = expected_counts[class_name]
            expected_tasks = expected_by_year_class.get((year, class_name), set())
            extracted_tasks = extracted_by_year_class.get((year, class_name), set())
            
            missing = expected_tasks - extracted_tasks
            extra = extracted_tasks - expected_tasks
            
            status_parts = []
            if len(extracted_tasks) == expected_count:
                status_parts.append("✓")
            elif len(missing) > 0:
                status_parts.append(f"⚠ {len(missing)} fehlen")
            
            if extra:
                status_parts.append(f"⚠ {len(extra)} extra")
            
            status = " ".join(status_parts) if status_parts else "✓"
            
            print(f"  {class_name:12s}: {len(extracted_tasks):2d}/{expected_count:2d}  {status}")
            
            if missing:
                missing_sorted = sorted(missing, key=lambda x: int(x))
                print(f"    Fehlend: {', '.join(missing_sorted[:10])}{' ...' if len(missing) > 10 else ''}")
            
            detail_data.append({
                'Jahr': year,
                'Klasse': class_name,
                'Extrahiert': len(extracted_tasks),
                'Erwartet': expected_count,
                'Fehlend': len(missing),
                'Fehlende_Aufgaben': ', '.join(sorted(missing, key=lambda x: int(x))) if missing else ''
            })
    
    print()
    print("=" * 80)
    print(f"GESAMT: {len(extracted)}/{len(solutions)} Aufgaben extrahiert")
    print(f"        {len(solutions) - len(extracted)} Aufgaben fehlen")
    print(f"        {100*len(extracted)/len(solutions):.1f}% Erfolgsrate")
    print("=" * 80)
    
    return results_data, detail_data


def export_to_excel(results_data, detail_data):
    """Exportiert Analyse-Ergebnisse nach Excel."""
    output_path = Path('analyse_1998_2011.xlsx')
    
    # Erstelle DataFrames
    df_overview = pd.DataFrame(results_data)
    df_detail = pd.DataFrame(detail_data)
    
    # Schreibe in Excel mit mehreren Sheets
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        df_overview.to_excel(writer, sheet_name='Übersicht', index=False)
        df_detail.to_excel(writer, sheet_name='Details', index=False)
    
    print(f"\n💾 Excel-Datei erstellt: {output_path}")
    print(f"   - Sheet 'Übersicht': Zusammenfassung nach Jahr")
    print(f"   - Sheet 'Details': Detaillierte Aufschlüsselung nach Jahr und Klasse")


def main():
    """Hauptfunktion."""
    results_data, detail_data = compare_extracted_vs_expected()
    export_to_excel(results_data, detail_data)


if __name__ == "__main__":
    main()
