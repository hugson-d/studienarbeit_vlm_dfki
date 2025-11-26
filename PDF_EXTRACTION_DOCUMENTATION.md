# PDF-Extraktions-Dokumentation

Diese Dokumentation beschreibt die technische Implementierung der beiden Extraktionsskripte für die Känguru-Wettbewerb Aufgaben.

## Übersicht

Das Projekt verwendet zwei verschiedene Extraktionsskripte, die auf die unterschiedlichen PDF-Formate der Jahrgänge abgestimmt sind:

1. **`extract_tasks_2012_2025.py`** - Direktextraktion für moderne PDFs (2012-2025)
2. **`extract_tasks_1998_2011.py`** - OCR-basierte Extraktion für ältere PDFs (1998-2011)

---

## 1. Extraktion für 2012-2025 (Direktextraktion)

### Verwendete Tools
- **PyMuPDF (fitz)**: PDF-Verarbeitung und Textextraktion
- **Python Pathlib**: Dateiverwaltung
- **Regex**: Dateinamen-Parsing

### Extraktionslogik

#### Phase 1: PDF-Analyse und Dateinamen-Parsing
```python
# Beispiel Dateiname: kaenguru2025_78.pdf
# → Jahr: 2025, Klasse: "7 und 8"
```

Das Skript parst den Dateinamen und mapped Klassencodes zu vollständigen Namen:
- `34` → "3 und 4"
- `56` → "5 und 6"
- `78` → "7 und 8"
- `910` → "9 und 10"
- `1113` → "11 bis 13"

#### Phase 2: Anchor-Detection (Aufgabenlabel-Erkennung)

Die moderne PDF-Struktur verwendet das Format **A1-A10, B1-B10, C1-C10** (bzw. A1-A8, B1-B8, C1-C8 für Klassen 3-4 und 5-6).

**Vorgehensweise:**
1. **Vollständige Textextraktion**: `page.get_text('words')` liest alle Wörter mit Positionen
2. **Anchor-Sammlung**: Suche nach allen Vorkommen der erwarteten Labels (A1, A2, B1, etc.)
3. **Position-Clustering**: 
   - Gruppierung aller gefundenen X-Positionen mit Toleranz von ±10 Punkten
   - Identifikation der häufigsten X-Position = Hauptspalte der Aufgabenlabels
4. **Deduplizierung**: Pro Label wird nur das erste Vorkommen verwendet

```python
# Beispiel: X-Position Clustering
x_tolerance = 10.0
# Alle Anchors bei X ≈ 42.5 werden gruppiert
# → Dies ist die Hauptspalte mit den Aufgabennummern
```

#### Phase 3: Vertikale Segmentierung

Für jede gefundene Aufgabe wird ein Bildbereich definiert:

**Vertikale Grenzen:**
- **Oben**: Label-Position minus 6pt Margin
- **Unten**: 
  - Falls nächste Aufgabe auf gleicher Seite: Nächstes Label minus 4pt
  - Falls letzte Aufgabe: Seitenende

**Horizontale Grenzen:**
- Links/Rechts: Seitenrand minus 6pt

```python
TOP, BOTTOM, SIDE = 6.0, 4.0, 6.0
clip = fitz.Rect(rect.x0 + SIDE, y_top, rect.x1 - SIDE, y_bot)
```

#### Phase 4: Image-Rendering

- **Auflösung**: 180 DPI (2.5x Vergrößerung von 72 DPI)
- **Format**: PNG ohne Alpha-Kanal
- **Dateiname**: `{year}_{class}_{label}.png` (z.B. `2025_7und8_A1.png`)

```python
zoom = 180 / 72  # 2.5x zoom
mat = fitz.Matrix(zoom, zoom)
pix = page.get_pixmap(matrix=mat, clip=clip, alpha=False)
```

### Beispiel-Workflow

```
kaenguru2025_78.pdf
    ↓ [Textextraktion]
    A1, A2, A3, ... B1, B2, ... C1, C2, ...
    ↓ [X-Clustering]
    Hauptspalte bei X=42.5 identifiziert
    ↓ [Deduplizierung]
    30 einzigartige Labels gefunden
    ↓ [Segmentierung & Rendering]
    30 PNG-Dateien erstellt (180 DPI)
```

---

## 2. Extraktion für 1998-2011 (OCR-basiert)

### Verwendete Tools
- **PyMuPDF (fitz)**: PDF-Verarbeitung
- **Tesseract OCR**: Texterkennung aus Bildern
- **pytesseract**: Python-Wrapper für Tesseract
- **Pillow (PIL)**: Bildverarbeitung
- **Regex**: Pattern-Matching für Aufgabennummern

### Problem: Warum OCR?

Die PDFs von 1998-2011 haben **Encoding-Probleme**:
- Direkte Textextraktion liefert unlesbaren Output (Non-Standard-Encoding)
- Beispiel: `\x00\x02\x01\n\x03\x05\x04\x07...` statt lesbarem Text

**Lösung:** Konvertierung zu Bildern und OCR-basierte Texterkennung

### Extraktionslogik

#### Phase 1: Marker-Detection (Aufgaben-Startpunkt)

**Herausforderung:** Unterschiedliche Terminologie zwischen Jahren:
- **1998**: "3-Punkte-Fragen" oder "6-Punkte-Fragen" (Klasse 3-4)
- **2000-2011**: "3-Punkte-Aufgaben"

**Vorgehensweise:**
1. **Seite zu Bild konvertieren** (2x Zoom für bessere OCR-Qualität)
2. **OCR mit Tesseract** (Deutsche Sprache: `lang='deu'`)
3. **Marker-Suche** mit mehreren Varianten:
   ```python
   markers = [
       '3-Punkte-Fragen',      # 1998 (die meisten Klassen)
       '6-Punkte-Fragen',      # 1998 (Klasse 3-4)
       '3-Punkte-Aufgaben',    # 2000-2011
       # + Varianten mit Leerzeichen
   ]
   ```
4. **Fuzzy Matching**: Bindestriche und Leerzeichen werden ignoriert
5. **Position-Rückrechnung**: Image-Koordinaten → PDF-Koordinaten (Division durch Zoom-Faktor)

```python
# OCR liefert Text-Position im 2x-gezoomten Bild
y_img = ocr_data['top'][i] + ocr_data['height'][i]
# Umrechnung in PDF-Koordinaten
y_pdf = y_img / zoom  # zoom = 2
```

#### Phase 2: Aufgabennummern-Extraktion via OCR

Die alte PDF-Struktur verwendet **numerische Labels**: 1. 2. 3. ... 21/30.

**Vorgehensweise:**
1. **Seitenweise OCR** ab der gefundenen Marker-Position
2. **Pattern-Matching**: Regex `^(\d+)\.$` findet "1.", "2.", etc.
3. **Koordinaten-Sammlung**: Alle gefundenen Nummern mit X/Y-Position
4. **Filterung**: Nur erwartete Nummern (1-21 für Klasse 3-4, 1-30 für andere)

```python
pattern = re.compile(r'^(\d+)\.$')
# Findet: "1." "2." "3." etc.
# Filtert aus: Seitenzahlen, andere Nummern
```

#### Phase 3: X-Position Clustering (identisch zu 2012-2025)

- Gruppierung aller X-Positionen mit ±10pt Toleranz
- Identifikation der Hauptspalte (häufigste Position)
- Deduplizierung: Erste Occurrence pro Label

#### Phase 4: Vertikale Segmentierung & Rendering

Identische Logik wie bei 2012-2025:
- Margins: TOP=6pt, BOTTOM=4pt, SIDE=6pt
- 180 DPI Rendering
- PNG-Output: `{year}_{class}_{number}.png` (z.B. `2009_7und8_15.png`)

### Besonderheiten 1998

**Problem Klasse 3-4 (1998):**
- Verwendet "6-Punkte-Fragen" statt "3-Punkte-Fragen"
- Nur 21 statt 24 Aufgaben erwartet

**Lösung im Code:**
```python
if class_code in ['3 und 4']:
    max_tasks = 21
else:
    max_tasks = 30
```

### Beispiel-Workflow (OCR)

```
kaenguru2009_78.pdf
    ↓ [Seite 1 → Bild @ 2x Zoom]
    ↓ [Tesseract OCR]
    "Känguruh 2009... 3-Punkte-Aufgaben ..."
    ↓ [Marker gefunden bei Y=206.0]
    ↓ [Alle Seiten → OCR]
    "1." "2." "3." ... "30." erkannt
    ↓ [X-Clustering]
    Hauptspalte bei X=42.5
    ↓ [Deduplizierung]
    30 einzigartige Nummern
    ↓ [Segmentierung @ 180 DPI]
    30 PNG-Dateien erstellt
```

---

## Gemeinsame Elemente

### 1. Image-Rendering Parameter
Beide Skripte verwenden identische Rendering-Einstellungen:
- **DPI**: 180 (optimal für Text-Lesbarkeit und Dateigröße)
- **Format**: PNG, 8-bit RGB (kein Alpha)
- **Zoom-Faktor**: 2.5x (180/72)

### 2. Margins und Clipping
```python
TOP = 6.0     # Abstand über dem Label
BOTTOM = 4.0  # Abstand zum nächsten Label
SIDE = 6.0    # Seitenränder links/rechts
```

Diese Werte wurden empirisch optimiert für:
- Vollständige Aufgabenerfassung
- Minimale Überlappung mit Nachbaraufgaben
- Entfernung von Seitenrändern und Kopf-/Fußzeilen

### 3. X-Position Clustering Algorithmus

**Problem:** Labels können leicht versetzt sein (Scan-Artefakte, OCR-Ungenauigkeiten)

**Lösung:** Clustering mit Toleranz
```python
x_tolerance = 10.0  # ±10 Punkte = ~3.5mm bei 72 DPI

# Gruppierung:
# X=42.3, X=42.8, X=43.1 → Cluster bei X≈42.5
# X=156.2, X=155.9 → separater Cluster (zu weit entfernt)
```

Der häufigste Cluster = Hauptspalte der Aufgabenlabels

### 4. Deduplizierungs-Strategie

**Problem:** Labels können mehrfach im Dokument vorkommen (Kopfzeilen, Inhaltsverzeichnisse)

**Lösung:** First-Occurrence-Prinzip
1. Sortierung nach Dokumentreihenfolge (Page → Y-Position)
2. Iteration mit Set-Tracking
3. Nur erste Occurrence wird behalten

```python
seen_labels = set()
for anchor in sorted_anchors:
    if anchor['label'] not in seen_labels:
        selected_anchors.append(anchor)  # ✓ Erste Occurrence
        seen_labels.add(anchor['label'])
```

---

## Performance-Metriken

### 2012-2025 (Direktextraktion)
- **Verarbeitungsgeschwindigkeit**: ~5-10 Sekunden pro PDF
- **Erfolgsrate**: ~100% (moderne PDFs sind konsistent)
- **Speicherbedarf**: ~50-100 KB pro Task-Image

### 1998-2011 (OCR-basiert)
- **Verarbeitungsgeschwindigkeit**: ~30-60 Sekunden pro PDF
- **Erfolgsrate**: ~95% (einige Marker-Detection-Fehler)
- **Speicherbedarf**: ~60-120 KB pro Task-Image

**Bekannte Fehlerquellen:**
- 1998: Teilweise fehlende Task-Nummern bei OCR-Erkennung
- 2001_11bis13: Marker nicht gefunden
- Einzelne Tasks können fehlen bei schlechter Scan-Qualität

---

## Technische Details

### PyMuPDF (fitz) Funktionen

**Textextraktion:**
```python
words = page.get_text('words')
# Liefert: [(x0, y0, x1, y1, word, block_no, line_no, word_no), ...]
```

**Pixmap-Rendering:**
```python
mat = fitz.Matrix(zoom, zoom)  # Skalierungsmatrix
pix = page.get_pixmap(matrix=mat, clip=rect, alpha=False)
png_bytes = pix.tobytes('png')
```

### Tesseract OCR Konfiguration

**Layout-Daten-Extraktion:**
```python
ocr_data = pytesseract.image_to_data(
    img, 
    lang='deu',  # Deutsche Sprache-Modell
    output_type=pytesseract.Output.DICT
)
# Liefert: {
#   'text': [...],      # Erkannter Text
#   'left': [...],      # X-Position
#   'top': [...],       # Y-Position
#   'width': [...],     # Breite
#   'height': [...]     # Höhe
# }
```

**OCR-Qualität:**
- **Zoom 2x**: Balance zwischen Genauigkeit und Performance
- **Deutsche Sprache**: Bessere Erkennung von Umlauten und Fachbegriffen
- **Layout-Mode**: Behält Positions-Informationen bei

---

## Zusammenfassung

| Aspekt | 2012-2025 | 1998-2011 |
|--------|-----------|-----------|
| **Methode** | Direktextraktion | OCR-basiert |
| **Tools** | PyMuPDF | PyMuPDF + Tesseract |
| **Label-Format** | A1-C10 (ABC) | 1-30 (numerisch) |
| **Marker** | Nicht nötig | "3-Punkte-Aufgaben/Fragen" |
| **Geschwindigkeit** | ~5-10s/PDF | ~30-60s/PDF |
| **Erfolgsrate** | ~100% | ~95% |
| **Hauptherausforderung** | - | PDF-Encoding, Marker-Varianten |

Beide Skripte folgen dem gleichen **Anchor-basierten Segmentierungs-Ansatz**:
1. Labels finden (direkt oder via OCR)
2. X-Position clustern → Hauptspalte identifizieren
3. Duplikate entfernen
4. Vertikale Bereiche berechnen
5. Als PNG rendern (180 DPI)

Die OCR-Lösung für 1998-2011 kompensiert erfolgreich die Encoding-Probleme der älteren PDFs, erfordert aber deutlich mehr Rechenzeit und ist anfälliger für Detektionsfehler.

---

## Struktur der Känguru-Aufgaben

### Aufgabenanzahl pro Jahrgangsstufe

Die Anzahl der Aufgaben variiert je nach **Jahr** und **Jahrgangsstufe**:

#### Zeitraum 1998-1999
| Jahrgangsstufe | Aufgabenanzahl | Besonderheit |
|----------------|----------------|--------------|
| **3-4** | 15 | Reduziert (statt 21/24) |
| **5-6** | 30 | Standard |
| **7-8** | 30 | Standard |
| **9-10** | 30 | Standard |
| **11-13** | 30 | Standard |

**Total pro Jahr:** 135 Aufgaben

#### Zeitraum 2000-2009
| Jahrgangsstufe | Aufgabenanzahl | Besonderheit |
|----------------|----------------|--------------|
| **3-4** | 21 | Erhöht auf 3×7 |
| **5-6** | 30 | Standard |
| **7-8** | 30 | Standard |
| **9-10** | 30 | Standard |
| **11-13** | 30 | Standard |

**Total pro Jahr:** 141 Aufgaben

#### Zeitraum 2010-2025
| Jahrgangsstufe | Aufgabenanzahl | Besonderheit |
|----------------|----------------|--------------|
| **3-4** | 24 | Standardisiert auf 3×8 |
| **5-6** | 24 | Angepasst auf 3×8 |
| **7-8** | 30 | Standard (3×10) |
| **9-10** | 30 | Standard (3×10) |
| **11-13** | 30 | Standard (3×10) |

**Total pro Jahr:** 138 Aufgaben

### Schwierigkeitsgrade (Originalformat)

Die Aufgaben sind in drei Schwierigkeitsstufen unterteilt, mit **unterschiedlicher Punktevergabe**:

#### 2012-2025 (Explizite ABC-Notation)

**Klassen 3-4 und 5-6:**
- **A-Aufgaben (Leicht):** 8 Aufgaben (A1-A8) → 3 Punkte je Aufgabe
- **B-Aufgaben (Mittel):** 8 Aufgaben (B1-B8) → 4 Punkte je Aufgabe
- **C-Aufgaben (Schwer):** 8 Aufgaben (C1-C8) → 5 Punkte je Aufgabe

**Klassen 7-8, 9-10 und 11-13:**
- **A-Aufgaben (Leicht):** 10 Aufgaben (A1-A10) → 3 Punkte je Aufgabe
- **B-Aufgaben (Mittel):** 10 Aufgaben (B1-B10) → 4 Punkte je Aufgabe
- **C-Aufgaben (Schwer):** 10 Aufgaben (C1-C10) → 5 Punkte je Aufgabe

**Verteilung:** Perfekt ausbalanciert (⅓ leicht, ⅓ mittel, ⅓ schwer)

#### 1998-2011 (Numerische Notation mit impliziter Schwierigkeit)

**Mapping-Regeln für 1998-2009:**

**Klasse 3-4 (21 Aufgaben):**
- **Leicht (A):** Aufgaben 1-7 → 3 Punkte
- **Mittel (B):** Aufgaben 8-14 → 4 Punkte
- **Schwer (C):** Aufgaben 15-21 → 5 Punkte

**Klasse 5-6, 7-8, 9-10, 11-13 (30 Aufgaben):**
- **Leicht (A):** Aufgaben 1-10 → 3 Punkte
- **Mittel (B):** Aufgaben 11-20 → 4 Punkte
- **Schwer (C):** Aufgaben 21-30 → 5 Punkte

**Mapping-Regeln für 2010-2011:**

**Klasse 3-4 und 5-6 (24 Aufgaben):**
- **Leicht (A):** Aufgaben 1-8 → 3 Punkte
- **Mittel (B):** Aufgaben 9-16 → 4 Punkte
- **Schwer (C):** Aufgaben 17-24 → 5 Punkte

**Klasse 7-8, 9-10, 11-13 (30 Aufgaben):**
- **Leicht (A):** Aufgaben 1-10 → 3 Punkte
- **Mittel (B):** Aufgaben 11-20 → 4 Punkte
- **Schwer (C):** Aufgaben 21-30 → 5 Punkte

**Verteilung:** Ebenfalls ausbalanciert (⅓ leicht, ⅓ mittel, ⅓ schwer)

### Punktevergabe und Wertung

**Originalwertung des Känguru-Wettbewerbs:**

| Schwierigkeit | Punkte bei richtiger Antwort | Abzug bei falscher Antwort |
|---------------|------------------------------|----------------------------|
| **A (Leicht)** | +3 Punkte | -0.75 Punkte (¼ Abzug) |
| **B (Mittel)** | +4 Punkte | -1.00 Punkte (¼ Abzug) |
| **C (Schwer)** | +5 Punkte | -1.25 Punkte (¼ Abzug) |

- **Startpunktzahl:** 30 Punkte (garantiertes Minimum bei Abgabe)
- **Keine Antwort:** 0 Punkte (kein Abzug)
- **Maximalpunktzahl:** Variable je nach Jahrgangsstufe
  - Klasse 3-4/5-6 (24 Aufgaben): 30 + 96 = **126 Punkte**
  - Klasse 7-8/9-10/11-13 (30 Aufgaben): 30 + 120 = **150 Punkte**

### Entwicklung der Aufgabenstruktur

**Chronologische Änderungen:**

1. **1998-1999**: Reduzierte Aufgabenanzahl für Klasse 3-4 (nur 15 statt 21)
   - Möglicherweise Testphase für jüngere Jahrgänge

2. **2000-2009**: Erhöhung auf 21 Aufgaben für Klasse 3-4
   - 3 Schwierigkeitsgrade à 7 Aufgaben (nicht perfekt ⅓)

3. **2010-2011**: Erste Standardisierung
   - Klasse 3-4 und 5-6: 24 Aufgaben (3×8)
   - Perfekte Drittelung

4. **2012-2025**: Einführung der ABC-Notation
   - Explizite Schwierigkeitskennzeichnung
   - Konsistente Struktur seit 13 Jahren
   - Klare Aufgabenlabels (A1-A8/A10, B1-B8/B10, C1-C8/C10)

### Besonderheiten einzelner Jahre

**1998 (Sonderformat):**
- Verwendet "Punkte-**Fragen**" statt "Punkte-**Aufgaben**"
- Klasse 3-4: Startet mit "**6-Punkte-Fragen**" (andere Klassen: 3-Punkte)
- Nur 15 Aufgaben für Klasse 3-4

**2010-2011 (Übergangsphase):**
- Anpassung der Aufgabenanzahl für Klassen 3-4 und 5-6
- Vorbereitung auf ABC-Format (aber noch numerisch)

Diese Strukturinformationen waren entscheidend für die **Mapping-Regeln** in `MAPPING_LOGIC.md` und die korrekte Zuordnung der Schwierigkeitsgrade in unserem Dataset.
