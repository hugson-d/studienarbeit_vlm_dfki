"""
Erstellt eine neue lösungen.json Datei aus der PDF kaenguru_loesungen_alle.pdf.

Dieses Skript liest die PDF-Datei aus und erstellt eine korrekte JSON-Datei
mit allen Lösungen.
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


def extract_solutions_from_pdf(pdf_path: Path) -> list:
    """Extrahiert alle Lösungen aus der PDF-Datei und gibt sie als Liste zurück."""
    print(f"\n📄 Lese PDF: {pdf_path}")
    
    doc = fitz.open(pdf_path)
    all_solutions = []
    
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
        
        if not current_year:
            print("(kein Jahr)")
            continue
        
        # Parsen der Tabellenstruktur
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
                        print(f"\n      → {current_class}", end="")
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
                            all_solutions.append({
                                "Jahr": current_year,
                                "Klasse": current_class,
                                "Aufgabe": aufgabe,
                                "Lösung": antwort
                            })
                    
                    continue
            
            i += 1
        
        print(f" ({len(all_solutions)} gesamt)")
    
    doc.close()
    
    print(f"\n✓ Extraktion abgeschlossen")
    return all_solutions


def save_solutions_to_json(solutions: list, output_path: Path):
    """Speichert die Lösungen in eine JSON-Datei."""
    print(f"\n💾 Speichere {len(solutions)} Lösungen nach: {output_path}")
    
    # Sortiere nach Jahr, Klasse, Aufgabe
    def sort_key(item):
        year = item['Jahr']
        klasse = item['Klasse']
        aufgabe = item['Aufgabe']
        
        # Extrahiere Kategorie und Nummer für Sortierung
        match = re.match(r'([ABC])(\d+)', aufgabe)
        if match:
            kategorie = match.group(1)
            nummer = int(match.group(2))
            return (year, klasse, kategorie, nummer)
        return (year, klasse, aufgabe, 0)
    
    solutions_sorted = sorted(solutions, key=sort_key)
    
    # Speichere als JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(solutions_sorted, f, ensure_ascii=False, indent=2)
    
    print("✓ Gespeichert!")


def print_statistics(solutions: list):
    """Gibt Statistiken über die extrahierten Lösungen aus."""
    print(f"\n{'='*80}")
    print("STATISTIKEN")
    print(f"{'='*80}")
    
    # Zähle nach Jahr und Klasse
    counts = defaultdict(lambda: defaultdict(int))
    for sol in solutions:
        counts[sol['Jahr']][sol['Klasse']] += 1
    
    print(f"\nGesamtzahl der Lösungen: {len(solutions)}")
    print(f"\nAnzahl pro Jahr und Klasse:")
    print("-" * 80)
    
    years = sorted(counts.keys())
    all_classes = sorted(set(k for year_data in counts.values() for k in year_data.keys()))
    
    # Header
    print(f"{'Jahr':<8}", end='')
    for klasse in all_classes:
        print(f"{klasse:<15}", end='')
    print("Gesamt")
    print("-" * 80)
    
    # Daten
    for year in years:
        print(f"{year:<8}", end='')
        year_total = 0
        for klasse in all_classes:
            count = counts[year].get(klasse, 0)
            if count > 0:
                print(f"{count:<15}", end='')
            else:
                print(f"{'-':<15}", end='')
            year_total += count
        print(f"{year_total}")
    
    # Gesamtsumme
    print("-" * 80)
    print(f"{'Gesamt':<8}", end='')
    for klasse in all_classes:
        total = sum(counts[year].get(klasse, 0) for year in years)
        print(f"{total:<15}", end='')
    grand_total = sum(sum(counts[year].values()) for year in years)
    print(f"{grand_total}")


def main():
    """Hauptfunktion."""
    project_root = Path(__file__).parent.parent
    pdf_path = project_root / 'kaenguru_loesungen_alle.pdf'
    output_path = project_root / 'lösungen.json'
    
    if not pdf_path.exists():
        print(f"❌ PDF nicht gefunden: {pdf_path}")
        return
    
    print("=" * 80)
    print("LÖSUNGEN.JSON NEU ERSTELLEN")
    print("=" * 80)
    
    # Extrahiere Lösungen aus PDF
    solutions = extract_solutions_from_pdf(pdf_path)
    
    # Zeige Statistiken
    print_statistics(solutions)
    
    # Speichere neue JSON-Datei
    save_solutions_to_json(solutions, output_path)
    
    print(f"\n{'='*80}")
    print("✅ FERTIG!")
    print(f"{'='*80}")
    print(f"\nDatei erstellt: {output_path}")
    print(f"\nBitte überprüfen Sie die Datei und führen Sie zur Kontrolle aus:")
    print(f"  python3 src/verify_solutions.py")
    print("=" * 80)


if __name__ == "__main__":
    main()
