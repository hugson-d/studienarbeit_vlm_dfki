# Source Code

# Source Code

Dieser Ordner enthält Python-Skripte für die Verarbeitung und Analyse der Känguru-Lösungen.

## Verfügbare Skripte

### 1. `create_solutions_from_pdf.py`
Erstellt die `lösungen.json` Datei aus der PDF `kaenguru_loesungen_alle.pdf`.

**Verwendung:**
```bash
python3 src/create_solutions_from_pdf.py
```

**Funktionen:**
- Extrahiert alle Lösungen von 1998-2025 aus der PDF
- Konvertiert alte Nummerierung (1-24/30) zu neuem Format (A1-C8/C10)
- Sortiert Lösungen nach Jahr, Klasse und Aufgabe
- Zeigt detaillierte Statistiken während der Extraktion

**Output:**
- Erstellt/überschreibt `lösungen.json` im Hauptverzeichnis
- 3741 Lösungen über 28 Jahre

---

### 2. `verify_solutions.py`
Verifiziert die `lösungen.json` Datei 3-fach gegen die PDF.

**Verwendung:**
```bash
python3 src/verify_solutions.py
```

**Funktionen:**
- Extrahiert Lösungen aus PDF
- Lädt lösungen.json
- Vergleicht beide Quellen in 3 Durchgängen
- Findet Fehler, Duplikate und fehlende Einträge

**Output:**
- Zeigt Anzahl korrekter Übereinstimmungen
- Listet alle gefundenen Fehler
- Speichert Fehlerbericht in `loesungen_fehler.json` (falls Fehler gefunden)

---

### 3. `analyze_solutions.py`
Analysiert die Lösungen statistisch.

**Verwendung:**
```bash
python3 src/analyze_solutions.py
```

**Funktionen:**
- Übersicht: Aufgaben pro Jahr und Klasse
- Detailanalyse: Alle Jahre einzeln mit Aufgabenliste
- Verteilung der Lösungsbuchstaben (A, B, C, D, E)
- Vergleich mit erwarteten Aufgabenzahlen

**Output:**
- Tabellen mit Aufgabenanzahlen
- Buchstabenverteilung pro Klassenstufe
- Hinweise auf fehlende oder unerwartete Aufgaben

---

## Datenformat

### lösungen.json
JSON-Array mit 3741 Einträgen:
```json
[
  {
    "Jahr": 1998,
    "Klasse": "3 und 4",
    "Aufgabe": "A1",
    "Lösung": "D"
  },
  ...
]
```

**Klassenstufen:**
- `"3 und 4"` (24 Aufgaben: A1-A8, B1-B8, C1-C8)
- `"5 und 6"` (24 Aufgaben: A1-A8, B1-B8, C1-C8)
- `"7 und 8"` (30 Aufgaben: A1-A10, B1-B10, C1-C10)
- `"9 und 10"` (30 Aufgaben: A1-A10, B1-B10, C1-C10)
- `"11 bis 13"` (30 Aufgaben: A1-A10, B1-B10, C1-C10)

**Hinweis:** Ältere Jahre (1998-2011) haben teilweise abweichende Aufgabenzahlen.

## Skripte

### `analyze_solutions.py`

Analysiert die `lösungen.json` Datei und gibt folgende Statistiken aus:

1. **Anzahl der Aufgaben pro Jahr und Klassenstufe** - Zeigt eine Tabelle mit allen Jahren und wie viele Aufgaben pro Klassenstufe vorhanden sind
2. **Verteilung der Lösungsbuchstaben** - Analysiert, wie oft jeder Buchstabe (A, B, C, D, E) als richtige Antwort vorkommt
3. **Test auf Gleichverteilung** - Prüft, ob die Lösungen gleichmäßig verteilt sind

**Verwendung:**
```bash
python3 src/analyze_solutions.py
```

## Erwartete Aufgabenstruktur

- **Klasse 3-4**: 24 Aufgaben (8 pro Kategorie A, B, C)
- **Klasse 5-6**: 24 Aufgaben (8 pro Kategorie A, B, C)
- **Klasse 7-8**: 30 Aufgaben (10 pro Kategorie A, B, C)
- **Klasse 9-10**: 30 Aufgaben (10 pro Kategorie A, B, C)
- **Klasse 11-13**: 30 Aufgaben (10 pro Kategorie A, B, C)

Ältere Jahre (vor 2010) können abweichende Strukturen haben.
