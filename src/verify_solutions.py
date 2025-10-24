"""
Überprüft die Lösungen aus lösungen.json gegen die PDF kaenguru_loesungen_alle.pdf.

Dieses Skript liest die PDF-Datei aus, extrahiert alle Lösungen und vergleicht sie
dreifach mit den Einträgen in lösungen.json um Fehler zu identifizieren.
"""

import json
import re
from pathlib import Path
from collections import defaultdict

try:
    import fitz  # PyMuPDF
except ImportError:
    print("❌ PyMuPDF ist nicht installiert. Bitte installieren Sie es mit:")
    print("   pip install PyMuPDF")
    exit(1)


def extract_solutions_from_pdf(pdf_path: Path) -> dict:
    """Extrahiert alle Lösungen aus der PDF-Datei."""
    print(f"\n📄 Lese PDF: {pdf_path}")
    
    doc = fitz.open(pdf_path)
    all_solutions = defaultdict(lambda: defaultdict(dict))
    
    print(f"\n🔍 Analysiere Seiten und extrahiere Lösungen...")
    
    for page_num, page in enumerate(doc):
        text = page.get_text()
        print(f"   Seite {page_num + 1}/{len(doc)}...", end=" ")
        
        lines = text.split('\n')
        
        current_year = None
        current_class = None
        
        # Suche nach Jahr in der Seite
        for line in lines:
            year_match = re.search(r'wettbewerb\s*(\d{4})|Jahr\s*(\d{4})|^(\d{4})$', line, re.IGNORECASE)
            if year_match:
                year_str = year_match.group(1) or year_match.group(2) or year_match.group(3)
                potential_year = int(year_str)
                if 1998 <= potential_year <= 2025:
                    current_year = potential_year
                    break
        
        # Parsen der Tabellenstruktur
        aufgaben_zeile = []
        antwort_zeile = []
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # Suche nach Klassenstufe
            if 'Klassenstufe' in line or 'Klasse' in line:
                class_patterns = [
                    (r'3\s*und\s*4', "3 und 4"),
                    (r'5\s*und\s*6', "5 und 6"),
                    (r'7\s*und\s*8', "7 und 8"),
                    (r'9\s*und\s*10', "9 und 10"),
                    (r'11\s*bis\s*13', "11 bis 13"),
                ]
                
                for pattern, class_name in class_patterns:
                    if re.search(pattern, line):
                        current_class = class_name
                        break
            
            # Wenn wir "Aufgabe" finden, sammle die Aufgabennummern
            if line == "Aufgabe" and i + 1 < len(lines):
                # Die nächste Zeile sollte die Aufgabennummern enthalten
                i += 1
                aufgaben_zeile = []
                # Sammle alle folgenden Zeilen bis "Antwort"
                while i < len(lines) and lines[i].strip() != "Antwort":
                    item = lines[i].strip()
                    # Neues Format: A1, B2, C3 (ab 2012)
                    if item and re.match(r'^[ABC]\d{1,2}$', item):
                        aufgaben_zeile.append(item)
                    # Altes Format: 1, 2, 3... (vor 2012)
                    elif item and re.match(r'^\d{1,2}$', item):
                        nummer = int(item)
                        # Konvertiere alte Nummerierung in neue (A1-A8, B9-B16, C17-C24 für Klasse 3-4/5-6)
                        # oder (A1-A10, B11-B20, C21-C30 für Klasse 7-8/9-10/11-13)
                        if current_class in ["3 und 4", "5 und 6"]:
                            if 1 <= nummer <= 8:
                                aufgaben_zeile.append(f"A{nummer}")
                            elif 9 <= nummer <= 16:
                                aufgaben_zeile.append(f"B{nummer - 8}")
                            elif 17 <= nummer <= 24:
                                aufgaben_zeile.append(f"C{nummer - 16}")
                        else:  # Klasse 7-8, 9-10, 11-13
                            if 1 <= nummer <= 10:
                                aufgaben_zeile.append(f"A{nummer}")
                            elif 11 <= nummer <= 20:
                                aufgaben_zeile.append(f"B{nummer - 10}")
                            elif 21 <= nummer <= 30:
                                aufgaben_zeile.append(f"C{nummer - 20}")
                    i += 1
                
                # Jetzt sollten die Antworten kommen
                if i < len(lines) and lines[i].strip() == "Antwort":
                    i += 1
                    antwort_zeile = []
                    # Sammle Antworten
                    while i < len(lines) and lines[i].strip() in ['A', 'B', 'C', 'D', 'E', 'F']:
                        antwort_zeile.append(lines[i].strip())
                        i += 1
                    
                    # Verknüpfe Aufgaben mit Antworten
                    if current_year and current_class and len(aufgaben_zeile) == len(antwort_zeile):
                        for aufgabe, antwort in zip(aufgaben_zeile, antwort_zeile):
                            all_solutions[current_year][current_class][aufgabe] = antwort
                    
                    continue
            
            i += 1
        
        count = sum(len(all_solutions[y][k]) for y in all_solutions for k in all_solutions[y])
        print(f"(Gesamt bisher: {count})")
    
    doc.close()
    
    print(f"\n✓ Extraktion abgeschlossen")
    return all_solutions


def load_json_solutions(json_path: Path) -> dict:
    """Lädt die Lösungen aus der JSON-Datei."""
    print(f"\n📋 Lade JSON: {json_path}")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        solutions_list = json.load(f)
    
    # Konvertiere in gleiche Struktur wie PDF
    solutions_dict = defaultdict(lambda: defaultdict(dict))
    
    for entry in solutions_list:
        year = entry['Jahr']
        klasse = entry['Klasse']
        aufgabe = entry['Aufgabe']
        loesung = entry['Lösung']
        
        solutions_dict[year][klasse][aufgabe] = loesung
    
    print(f"✓ {len(solutions_list)} Einträge geladen")
    return solutions_dict


def compare_solutions(pdf_solutions: dict, json_solutions: dict, round_num: int):
    """Vergleicht die Lösungen aus PDF und JSON."""
    print(f"\n{'='*80}")
    print(f"VERGLEICH RUNDE {round_num}/3")
    print(f"{'='*80}")
    
    errors = []
    missing_in_json = []
    missing_in_pdf = []
    correct_count = 0
    
    # Prüfe alle PDF-Lösungen
    for year in sorted(pdf_solutions.keys()):
        for klasse in sorted(pdf_solutions[year].keys()):
            for aufgabe in sorted(pdf_solutions[year][klasse].keys()):
                pdf_solution = pdf_solutions[year][klasse][aufgabe]
                json_solution = json_solutions.get(year, {}).get(klasse, {}).get(aufgabe)
                
                if json_solution is None:
                    missing_in_json.append({
                        'Jahr': year,
                        'Klasse': klasse,
                        'Aufgabe': aufgabe,
                        'Lösung_PDF': pdf_solution
                    })
                elif json_solution != pdf_solution:
                    errors.append({
                        'Jahr': year,
                        'Klasse': klasse,
                        'Aufgabe': aufgabe,
                        'Lösung_PDF': pdf_solution,
                        'Lösung_JSON': json_solution
                    })
                else:
                    correct_count += 1
    
    # Prüfe auf Einträge die in JSON aber nicht in PDF sind
    for year in sorted(json_solutions.keys()):
        for klasse in sorted(json_solutions[year].keys()):
            for aufgabe in sorted(json_solutions[year][klasse].keys()):
                if aufgabe not in pdf_solutions.get(year, {}).get(klasse, {}):
                    missing_in_pdf.append({
                        'Jahr': year,
                        'Klasse': klasse,
                        'Aufgabe': aufgabe,
                        'Lösung_JSON': json_solutions[year][klasse][aufgabe]
                    })
    
    # Ergebnisse ausgeben
    print(f"\n✓ Korrekte Übereinstimmungen: {correct_count}")
    
    if errors:
        print(f"\n❌ FEHLER gefunden: {len(errors)}")
        print(f"\nFehlerhafte Einträge (PDF ≠ JSON):")
        print("-" * 80)
        for error in errors[:50]:  # Erste 50 Fehler
            print(f"  Jahr {error['Jahr']}, {error['Klasse']}, {error['Aufgabe']}:")
            print(f"    PDF:  {error['Lösung_PDF']}")
            print(f"    JSON: {error['Lösung_JSON']}")
        if len(errors) > 50:
            print(f"  ... und {len(errors) - 50} weitere Fehler")
    
    if missing_in_json:
        print(f"\n⚠️  In PDF aber nicht in JSON: {len(missing_in_json)}")
        for missing in missing_in_json[:20]:
            print(f"  Jahr {missing['Jahr']}, {missing['Klasse']}, {missing['Aufgabe']} → {missing['Lösung_PDF']}")
        if len(missing_in_json) > 20:
            print(f"  ... und {len(missing_in_json) - 20} weitere")
    
    if missing_in_pdf:
        print(f"\n⚠️  In JSON aber nicht in PDF: {len(missing_in_pdf)}")
        for missing in missing_in_pdf[:20]:
            print(f"  Jahr {missing['Jahr']}, {missing['Klasse']}, {missing['Aufgabe']} → {missing['Lösung_JSON']}")
        if len(missing_in_pdf) > 20:
            print(f"  ... und {len(missing_in_pdf) - 20} weitere")
    
    return errors, missing_in_json, missing_in_pdf


def main():
    """Hauptfunktion."""
    project_root = Path(__file__).parent.parent
    pdf_path = project_root / 'kaenguru_loesungen_alle.pdf'
    json_path = project_root / 'lösungen.json'
    
    if not pdf_path.exists():
        print(f"❌ PDF nicht gefunden: {pdf_path}")
        return
    
    if not json_path.exists():
        print(f"❌ JSON nicht gefunden: {json_path}")
        return
    
    print("=" * 80)
    print("LÖSUNGEN VERIFIKATION - 3-FACH PRÜFUNG")
    print("=" * 80)
    
    # Extrahiere Lösungen aus PDF
    pdf_solutions = extract_solutions_from_pdf(pdf_path)
    
    # Lade JSON-Lösungen
    json_solutions = load_json_solutions(json_path)
    
    # Zeige Statistik
    pdf_count = sum(len(pdf_solutions[y][k]) for y in pdf_solutions for k in pdf_solutions[y])
    json_count = sum(len(json_solutions[y][k]) for y in json_solutions for k in json_solutions[y])
    
    print(f"\n📊 Statistik:")
    print(f"   PDF:  {pdf_count} Lösungen")
    print(f"   JSON: {json_count} Lösungen")
    
    # Führe 3 Vergleiche durch
    all_errors = []
    for round_num in range(1, 4):
        errors, missing_in_json, missing_in_pdf = compare_solutions(
            pdf_solutions, json_solutions, round_num
        )
        all_errors.append((errors, missing_in_json, missing_in_pdf))
    
    # Finale Zusammenfassung
    print(f"\n{'='*80}")
    print("FINALE ZUSAMMENFASSUNG (nach 3 Durchgängen)")
    print(f"{'='*80}")
    
    # Prüfe ob alle 3 Durchgänge gleiche Ergebnisse haben
    if all(len(all_errors[0][0]) == len(e[0]) for e in all_errors):
        print("✓ Alle 3 Durchgänge zeigen konsistente Ergebnisse")
    else:
        print("⚠️ Inkonsistente Ergebnisse zwischen den Durchgängen!")
    
    errors, missing_in_json, missing_in_pdf = all_errors[0]
    
    if not errors and not missing_in_json and not missing_in_pdf:
        print("\n✅ PERFEKT! Alle Lösungen wurden korrekt überführt!")
        print("   Keine Fehler gefunden.")
    else:
        print(f"\n❌ PROBLEME GEFUNDEN:")
        if errors:
            print(f"   - {len(errors)} fehlerhafte Übertragungen (PDF ≠ JSON)")
        if missing_in_json:
            print(f"   - {len(missing_in_json)} Lösungen in PDF aber nicht in JSON")
        if missing_in_pdf:
            print(f"   - {len(missing_in_pdf)} Lösungen in JSON aber nicht in PDF")
        
        # Speichere Fehler in Datei
        if errors or missing_in_json or missing_in_pdf:
            error_file = project_root / 'loesungen_fehler.json'
            with open(error_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'fehlerhafte_uebertagungen': errors,
                    'fehlt_in_json': missing_in_json,
                    'fehlt_in_pdf': missing_in_pdf
                }, f, ensure_ascii=False, indent=2)
            print(f"\n💾 Details gespeichert in: {error_file}")
    
    print("=" * 80)


if __name__ == "__main__":
    main()
