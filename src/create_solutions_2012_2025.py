"""
Erstellt lösungen_2012_2025.json aus der Lösungs-PDF für Jahre 2012-2025.

Diese PDFs verwenden das Format A1-A10, B1-B10, C1-C10 (bzw. A1-A8, B1-B8, C1-C8 für Klasse 3-4/5-6).
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


def extract_solutions_2012_2025(pdf_path: Path) -> list:
    """Extrahiert Lösungen für Jahre 2012-2025 aus der PDF-Datei."""
    print(f"\n📄 Lese PDF: {pdf_path}")
    
    doc = fitz.open(pdf_path)
    all_solutions = []
    
    print(f"\n🔍 Analysiere Seiten und extrahiere Lösungen (2012-2025)...")
    
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
                # Nur Jahre 2012-2025
                if 2012 <= potential_year <= 2025:
                    current_year = potential_year
                    break
        
        if not current_year:
            print("(kein Jahr oder < 2012)")
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
                    # Format: A1, B2, C3 (ab 2012)
                    if item and re.match(r'^[ABC]\d{1,2}$', item):
                        aufgaben_zeile.append(item)
                    i += 1
                
                # Jetzt sammle die Antworten
                if i < len(lines) and lines[i].strip() == "Antwort":
                    i += 1
                    antworten_zeile = []
                    # Sammle alle Antworten (A, B, C, D, E)
                    collected = 0
                    while i < len(lines) and collected < len(aufgaben_zeile):
                        item = lines[i].strip()
                        if item in ['A', 'B', 'C', 'D', 'E']:
                            antworten_zeile.append(item)
                            collected += 1
                        i += 1
                    
                    # Kombiniere Aufgaben und Antworten
                    if len(aufgaben_zeile) == len(antworten_zeile) and current_year and current_class:
                        for aufgabe, antwort in zip(aufgaben_zeile, antworten_zeile):
                            all_solutions.append({
                                "Jahr": current_year,
                                "Klasse": current_class,
                                "Aufgabe": aufgabe,
                                "Lösung": antwort
                            })
                        print(f" ({len(aufgaben_zeile)} Aufgaben)", end="")
            
            i += 1
        
        print()  # Neue Zeile nach Seite
    
    doc.close()
    
    # Sortiere nach Jahr, Klasse, Aufgabe
    all_solutions.sort(key=lambda x: (
        x["Jahr"],
        ["3 und 4", "5 und 6", "7 und 8", "9 und 10", "11 bis 13"].index(x["Klasse"]),
        x["Aufgabe"]
    ))
    
    return all_solutions


def main():
    """Hauptfunktion."""
    # Pfade
    pdf_path = Path("data/references/kaenguru_loesungen_alle.pdf")
    output_path = Path("lösungen_2012_2025.json")
    
    if not pdf_path.exists():
        print(f"❌ PDF-Datei nicht gefunden: {pdf_path}")
        return
    
    # Extrahiere Lösungen
    solutions = extract_solutions_2012_2025(pdf_path)
    
    # Statistiken
    print(f"\n📊 Statistiken:")
    print(f"   Gesamt: {len(solutions)} Lösungen")
    
    years = defaultdict(int)
    classes = defaultdict(int)
    for sol in solutions:
        years[sol["Jahr"]] += 1
        classes[sol["Klasse"]] += 1
    
    print(f"\n   Nach Jahr:")
    for year in sorted(years.keys()):
        if 2012 <= year <= 2025:
            print(f"      {year}: {years[year]} Lösungen")
    
    print(f"\n   Nach Klasse:")
    for class_name in ["3 und 4", "5 und 6", "7 und 8", "9 und 10", "11 bis 13"]:
        if class_name in classes:
            print(f"      {class_name}: {classes[class_name]} Lösungen")
    
    # Speichere als JSON
    print(f"\n💾 Speichere in: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(solutions, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Fertig! {len(solutions)} Lösungen gespeichert.")


if __name__ == "__main__":
    main()
