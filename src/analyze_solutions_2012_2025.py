"""
Analyse der Känguru-Lösungen und extrahierten Tasks (2012-2025).

Dieses Skript analysiert:
1. Die lösungen_2012_2025.json Datei
2. Die extrahierten Tasks aus tasks_manifest_2012_2025.jsonl
3. Vergleicht Soll- vs. Ist-Zahlen
4. Verteilung der Lösungsbuchstaben (A, B, C, D, E)
5. Exportiert Ergebnisse nach Excel
"""

import json
from pathlib import Path
from collections import defaultdict, Counter
import pandas as pd


def load_solutions(filepath: Path) -> list:
    """Lädt die Lösungen aus der lösungen_2012_2025.json Datei."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_extracted_tasks(filepath: Path) -> list:
    """Lädt die extrahierten Tasks aus der JSONL-Datei."""
    tasks = []
    if not filepath.exists():
        return tasks
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            tasks.append(json.loads(line))
    
    return tasks


def analyze_tasks_per_year_and_class(solutions: list):
    """Analysiert die Anzahl der Aufgaben pro Jahr und Klassenstufe."""
    # Gruppiere nach Jahr und Klasse
    tasks_count = defaultdict(lambda: defaultdict(int))
    
    for entry in solutions:
        jahr = entry['Jahr']
        klasse = entry['Klasse']
        tasks_count[jahr][klasse] += 1
    
    return tasks_count


def analyze_solution_distribution(solutions: list):
    """Analysiert die Verteilung der Lösungsbuchstaben."""
    # Gesamtverteilung
    overall_counter = Counter()
    
    # Verteilung pro Klassenstufe
    by_class = defaultdict(Counter)
    
    # Verteilung pro Jahr
    by_year = defaultdict(Counter)
    
    for entry in solutions:
        lösung = entry['Lösung']
        klasse = entry['Klasse']
        jahr = entry['Jahr']
        
        overall_counter[lösung] += 1
        by_class[klasse][lösung] += 1
        by_year[jahr][lösung] += 1
    
    return overall_counter, by_class, by_year


def print_tasks_statistics(tasks_count: dict):
    """Gibt die Aufgabenstatistiken aus."""
    print("=" * 80)
    print("ANZAHL DER AUFGABEN PRO JAHR UND KLASSENSTUFE")
    print("=" * 80)
    
    # Sortiere Jahre
    years = sorted(tasks_count.keys())
    
    # Alle Klassenstufen sammeln
    all_classes = set()
    for year_data in tasks_count.values():
        all_classes.update(year_data.keys())
    all_classes = sorted(all_classes)
    
    # Header
    print(f"\n{'Jahr':<8}", end='')
    for klasse in all_classes:
        print(f"{klasse:<15}", end='')
    print("Gesamt")
    print("-" * 80)
    
    # Daten
    total_per_class = defaultdict(int)
    grand_total = 0
    
    for year in years:
        print(f"{year:<8}", end='')
        year_total = 0
        for klasse in all_classes:
            count = tasks_count[year].get(klasse, 0)
            if count > 0:
                print(f"{count:<15}", end='')
            else:
                print(f"{'-':<15}", end='')
            year_total += count
            total_per_class[klasse] += count
        print(f"{year_total}")
        grand_total += year_total
    
    # Summen
    print("-" * 80)
    print(f"{'Gesamt':<8}", end='')
    for klasse in all_classes:
        print(f"{total_per_class[klasse]:<15}", end='')
    print(f"{grand_total}")
    print()


def print_solution_distribution(overall: Counter, by_class: dict, by_year: dict):
    """Gibt die Verteilung der Lösungsbuchstaben aus."""
    print("=" * 80)
    print("VERTEILUNG DER LÖSUNGSBUCHSTABEN")
    print("=" * 80)
    
    # Gesamtverteilung
    print("\n1. Gesamtverteilung:")
    print("-" * 40)
    total = sum(overall.values())
    letters = sorted(overall.keys())
    
    for letter in letters:
        count = overall[letter]
        percentage = (count / total * 100) if total > 0 else 0
        bar = '█' * int(percentage / 2)
        print(f"{letter}: {count:4d} ({percentage:5.2f}%) {bar}")
    print(f"\nGesamt: {total} Lösungen")
    
    # Verteilung pro Klassenstufe
    print("\n2. Verteilung pro Klassenstufe:")
    print("-" * 40)
    for klasse in sorted(by_class.keys()):
        counter = by_class[klasse]
        total_class = sum(counter.values())
        print(f"\n{klasse}:")
        for letter in sorted(counter.keys()):
            count = counter[letter]
            percentage = (count / total_class * 100) if total_class > 0 else 0
            print(f"  {letter}: {count:4d} ({percentage:5.2f}%)")
    
    # Statistische Analyse: Chi-Quadrat Test für Gleichverteilung
    print("\n3. Test auf Gleichverteilung:")
    print("-" * 40)
    expected_percentage = 20.0  # Bei Gleichverteilung sollte jeder Buchstabe 20% haben
    print(f"Bei perfekter Gleichverteilung: 20% pro Buchstabe")
    print(f"\nAbweichungen von der Gleichverteilung:")
    for letter in letters:
        count = overall[letter]
        percentage = (count / total * 100) if total > 0 else 0
        deviation = percentage - expected_percentage
        status = "✓" if abs(deviation) < 2 else "⚠"
        print(f"  {status} {letter}: {percentage:5.2f}% (Abweichung: {deviation:+.2f}%)")


def analyze_tasks_per_year_detailed(solutions: list):
    """Analysiert die Aufgaben detailliert pro Jahr und prüft auf Vollständigkeit."""
    # Erwartete Anzahl pro Klassenstufe (ab 2012)
    expected = {
        "3 und 4": 24,
        "5 und 6": 24,
        "7 und 8": 30,
        "9 und 10": 30,
        "11 bis 13": 30
    }
    
    # Gruppiere nach Jahr und Klasse
    tasks_by_year_class = defaultdict(lambda: defaultdict(list))
    
    for entry in solutions:
        jahr = entry['Jahr']
        klasse = entry['Klasse']
        aufgabe = entry['Aufgabe']
        tasks_by_year_class[jahr][klasse].append(aufgabe)
    
    print("\n" + "=" * 80)
    print("DETAILLIERTE ANALYSE PRO JAHR")
    print("=" * 80)
    
    years = sorted(tasks_by_year_class.keys())
    
    for year in years:
        print(f"\n{'─' * 80}")
        print(f"Jahr {year}:")
        print(f"{'─' * 80}")
        
        year_data = tasks_by_year_class[year]
        all_classes = sorted(year_data.keys())
        
        year_total = 0
        has_issues = False
        
        for klasse in all_classes:
            aufgaben = sorted(year_data[klasse])
            count = len(aufgaben)
            year_total += count
            
            # Prüfe ob Anzahl stimmt (ab 2012)
            expected_count = expected.get(klasse, None)
            status = ""
            
            if expected_count and year >= 2012:
                if count == expected_count:
                    status = "✓"
                elif count < expected_count:
                    status = f"⚠ FEHLT: {expected_count - count}"
                    has_issues = True
                else:
                    status = f"⚠ ZU VIEL: +{count - expected_count}"
                    has_issues = True
            elif expected_count and count != expected_count:
                status = f"({expected_count} erwartet)"
            
            # Zeige Aufgabenliste
            aufgaben_str = ', '.join(aufgaben[:10])
            if len(aufgaben) > 10:
                aufgaben_str += f", ... ({len(aufgaben) - 10} weitere)"
            
            print(f"  {klasse:<15} {count:2d} Aufgaben  {status}")
            print(f"    └─ {aufgaben_str}")
        
        print(f"\n  Gesamt {year}: {year_total} Aufgaben")
        if has_issues:
            print(f"  ⚠ Dieses Jahr hat Abweichungen!")
    
    print(f"\n{'─' * 80}")


def print_missing_tasks_info():
    """Gibt Informationen über erwartete Aufgabenanzahlen aus."""
    print("\n" + "=" * 80)
    print("ERWARTETE AUFGABENANZAHLEN (2012-2025)")
    print("=" * 80)
    print("""
Aufgabenstruktur beim Känguru-Wettbewerb:

Klasse 3 und 4:   24 Aufgaben (A1-A8, B1-B8, C1-C8)
Klasse 5 und 6:   24 Aufgaben (A1-A8, B1-B8, C1-C8)
Klasse 7 und 8:   30 Aufgaben (A1-A10, B1-B10, C1-C10)
Klasse 9 und 10:  30 Aufgaben (A1-A10, B1-B10, C1-C10)
Klasse 11 bis 13: 30 Aufgaben (A1-A10, B1-B10, C1-C10)

Pro Jahr: 138 Aufgaben gesamt
Zeitraum 2012-2025: 14 Jahre × 138 = 1932 Aufgaben
""")


def compare_extracted_vs_expected(extracted_tasks: list, solutions: list):
    """Vergleicht extrahierte Tasks mit erwarteten Lösungen."""
    print("\n" + "=" * 80)
    print("VERGLEICH: EXTRAHIERTE TASKS vs. LÖSUNGEN (2012-2025)")
    print("=" * 80)
    
    # Gruppiere extrahierte Tasks
    extracted_count = defaultdict(lambda: defaultdict(int))
    for task in extracted_tasks:
        extracted_count[task['year']][task['class']] += 1
    
    # Gruppiere Lösungen
    solutions_count = defaultdict(lambda: defaultdict(int))
    for entry in solutions:
        solutions_count[entry['Jahr']][entry['Klasse']] += 1
    
    # Erwartete Zahlen
    expected = {
        "3 und 4": 24,
        "5 und 6": 24,
        "7 und 8": 30,
        "9 und 10": 30,
        "11 bis 13": 30
    }
    
    all_years = sorted(set(list(extracted_count.keys()) + list(solutions_count.keys())))
    all_classes = sorted(expected.keys())
    
    print(f"\n{'Jahr':<8} {'Klasse':<15} {'Erwartet':>10} {'Lösungen':>10} {'Extrahiert':>12} {'Status':>10}")
    print("-" * 80)
    
    total_expected = 0
    total_solutions = 0
    total_extracted = 0
    
    for year in all_years:
        for i, klasse in enumerate(all_classes):
            exp = expected[klasse]
            sol = solutions_count[year].get(klasse, 0)
            ext = extracted_count[year].get(klasse, 0)
            
            # Status
            if ext == exp == sol:
                status = "✓"
            elif ext == exp and sol == exp:
                status = "✓"
            elif ext < exp:
                status = f"⚠ -{exp-ext}"
            else:
                status = "⚠"
            
            year_str = str(year) if i == 0 else ""
            print(f"{year_str:<8} {klasse:<15} {exp:>10} {sol:>10} {ext:>12} {status:>10}")
            
            total_expected += exp
            total_solutions += sol
            total_extracted += ext
        
        if year < max(all_years):
            print()
    
    print("-" * 80)
    print(f"{'GESAMT':<8} {'2012-2025':<15} {total_expected:>10} {total_solutions:>10} {total_extracted:>12}")
    print()
    
    # Zusammenfassung
    if total_extracted == total_expected:
        print("✓ Alle erwarteten Tasks wurden extrahiert!")
    else:
        missing = total_expected - total_extracted
        print(f"⚠ {missing} Tasks fehlen noch (von {total_expected} erwartet)")


def main():
    """Hauptfunktion."""
    # Pfade
    project_root = Path(__file__).parent.parent
    solutions_file = project_root / 'lösungen_2012_2025.json'
    manifest_file = project_root / 'data' / 'references' / 'tasks_manifest_2012_2025.jsonl'
    
    if not solutions_file.exists():
        print(f"❌ Fehler: Datei nicht gefunden: {solutions_file}")
        return
    
    print("\n" + "=" * 80)
    print("KÄNGURU-ANALYSE: 2012-2025")
    print("=" * 80)
    print(f"Lösungen: {solutions_file}")
    print(f"Manifest: {manifest_file}")
    
    # Lade Lösungen (nur 2012-2025)
    solutions = load_solutions(solutions_file)
    print(f"\nLösungen geladen: {len(solutions)} (Jahre 2012-2025)")
    
    # Lade extrahierte Tasks
    extracted_tasks = load_extracted_tasks(manifest_file)
    print(f"Extrahierte Tasks: {len(extracted_tasks)}")
    
    # Vergleiche extrahierte vs. erwartete
    if extracted_tasks:
        compare_extracted_vs_expected(extracted_tasks, solutions)
    
    # Analysiere Aufgaben pro Jahr und Klasse
    tasks_count = analyze_tasks_per_year_and_class(solutions)
    print_tasks_statistics(tasks_count)
    
    # Detaillierte Analyse pro Jahr
    analyze_tasks_per_year_detailed(solutions)
    
    # Analysiere Lösungsverteilung
    overall, by_class, by_year = analyze_solution_distribution(solutions)
    print_solution_distribution(overall, by_class, by_year)
    
    # Zeige erwartete Aufgabenanzahlen
    print_missing_tasks_info()
    
    # Exportiere nach Excel
    export_to_excel(solutions, extracted_tasks)
    
    print("=" * 80)
    print("✓ Analyse abgeschlossen")
    print("=" * 80 + "\n")


def export_to_excel(solutions: list, extracted_tasks: list):
    """Exportiert Analyse-Ergebnisse nach Excel."""
    output_path = Path('analyse_2012_2025.xlsx')
    
    # 1. Übersicht nach Jahr
    years_data = []
    tasks_by_year_class = defaultdict(lambda: defaultdict(set))
    extracted_by_year_class = defaultdict(lambda: defaultdict(set))
    
    for sol in solutions:
        tasks_by_year_class[sol['Jahr']][sol['Klasse']].add(sol['Aufgabe'])
    
    for task in extracted_tasks:
        extracted_by_year_class[task['year']][task['class']].add(task['task_id'])
    
    expected_counts = {"3 und 4": 24, "5 und 6": 24, "7 und 8": 30, "9 und 10": 30, "11 bis 13": 30}
    
    for year in sorted(set(sol['Jahr'] for sol in solutions)):
        year_expected = sum(expected_counts[cls] for cls in expected_counts.keys())
        year_extracted = sum(len(extracted_by_year_class[year][cls]) for cls in expected_counts.keys())
        
        years_data.append({
            'Jahr': year,
            'Extrahiert': year_extracted,
            'Erwartet': year_expected,
            'Fehlend': year_expected - year_extracted,
            'Prozent': f"{100*year_extracted/year_expected:.1f}%" if year_expected > 0 else "0%"
        })
    
    # 2. Details nach Jahr und Klasse
    details_data = []
    for year in sorted(set(sol['Jahr'] for sol in solutions)):
        for class_name in ['3 und 4', '5 und 6', '7 und 8', '9 und 10', '11 bis 13']:
            expected = expected_counts[class_name]
            extracted = extracted_by_year_class[year][class_name]
            missing = tasks_by_year_class[year][class_name] - extracted
            
            details_data.append({
                'Jahr': year,
                'Klasse': class_name,
                'Extrahiert': len(extracted),
                'Erwartet': expected,
                'Fehlend': len(missing),
                'Fehlende_Aufgaben': ', '.join(sorted(missing)) if missing else ''
            })
    
    # 3. Lösungsverteilung
    distribution_data = []
    overall_counter = Counter(sol['Lösung'] for sol in solutions)
    total = sum(overall_counter.values())
    
    for letter in sorted(overall_counter.keys()):
        count = overall_counter[letter]
        percentage = (count / total * 100) if total > 0 else 0
        
        distribution_data.append({
            'Lösung': letter,
            'Anzahl': count,
            'Prozent': f"{percentage:.2f}%",
            'Abweichung_von_20%': f"{percentage - 20:.2f}%"
        })
    
    # Schreibe in Excel
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        pd.DataFrame(years_data).to_excel(writer, sheet_name='Übersicht', index=False)
        pd.DataFrame(details_data).to_excel(writer, sheet_name='Details', index=False)
        pd.DataFrame(distribution_data).to_excel(writer, sheet_name='Lösungsverteilung', index=False)
    
    print(f"\n💾 Excel-Datei erstellt: {output_path}")
    print(f"   - Sheet 'Übersicht': Zusammenfassung nach Jahr")
    print(f"   - Sheet 'Details': Detaillierte Aufschlüsselung nach Jahr und Klasse")
    print(f"   - Sheet 'Lösungsverteilung': Verteilung der Lösungsbuchstaben A-E")


if __name__ == "__main__":
    main()
