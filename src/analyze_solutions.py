"""
Analyse der Känguru-Lösungen.

Dieses Skript analysiert die lösungen.json Datei und gibt folgende Statistiken aus:
1. Anzahl der Aufgaben pro Jahr und Klassenstufe
2. Verteilung der Lösungsbuchstaben (A, B, C, D, E)
"""

import json
from pathlib import Path
from collections import defaultdict, Counter


def load_solutions(filepath: Path) -> list:
    """Lädt die Lösungen aus der JSON-Datei."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


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
    print("ERWARTETE AUFGABENANZAHLEN")
    print("=" * 80)
    print("""
Typische Aufgabenstruktur beim Känguru-Wettbewerb:

Klasse 3-4:    24 Aufgaben (A1-A8, B1-B8, C1-C8)
Klasse 5-6:    24 Aufgaben (A1-A8, B1-B8, C1-C8)
Klasse 7-8:    30 Aufgaben (A1-A10, B1-B10, C1-C10)
Klasse 9-10:   30 Aufgaben (A1-A10, B1-B10, C1-C10)
Klasse 11-13:  30 Aufgaben (A1-A10, B1-B10, C1-C10)

Hinweis: Die Struktur kann in älteren Jahren abweichen.
Vergleichen Sie die obige Tabelle mit diesen Erwartungen.
""")


def main():
    """Hauptfunktion."""
    # Pfade
    project_root = Path(__file__).parent.parent
    solutions_file = project_root / 'lösungen.json'
    
    if not solutions_file.exists():
        print(f"❌ Fehler: Datei nicht gefunden: {solutions_file}")
        return
    
    print("\n" + "=" * 80)
    print("KÄNGURU-LÖSUNGEN ANALYSE")
    print("=" * 80)
    print(f"Datei: {solutions_file}")
    
    # Lade Lösungen
    solutions = load_solutions(solutions_file)
    print(f"Geladene Einträge: {len(solutions)}")
    
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
    
    print("=" * 80)
    print("✓ Analyse abgeschlossen")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
