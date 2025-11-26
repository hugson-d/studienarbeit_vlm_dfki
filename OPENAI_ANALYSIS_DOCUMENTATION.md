# OpenAI-basierte Dataset-Analyse Pipeline

## Übersicht

Dieses Dokument beschreibt die drei OpenAI-basierten Skripte zur semantischen Anreicherung des Känguru-Mathematik-Datasets. Diese Skripte nutzen die OpenAI Vision API (GPT-4o und GPT-4o-mini), um zusätzliche Metadaten zu extrahieren, die durch reine PDF-Extraktion nicht verfügbar sind.

### Pipeline-Übersicht

```
dataset_final.json (Basis-Daten)
    ↓
1. categorize_math_tasks.py  → Fügt "math_category" hinzu
    ↓
2. extract_text.py           → Fügt "extracted_text" hinzu
    ↓
3. analyze_text_only.py      → Fügt "is_text_only" hinzu
    ↓
dataset_final.json (Vollständig angereichert)
```

### Zweck der Pipeline

Die drei Skripte ermöglichen:
- **Fachbereichs-spezifische Evaluation**: VLM-Performance nach mathematischen Teilbereichen analysieren
- **Text-Only vs. Multimodal**: Vergleich der Modell-Leistung bei reinen Textaufgaben vs. visuell-anspruchsvollen Aufgaben
- **Textuelle Analyse**: Verwendung der extrahierten Fragen/Antworten für NLP-basierte Untersuchungen
- **Visuelles Verständnis**: Identifikation von Aufgaben, die echtes "Sehen" erfordern vs. reines "Lesen"

---

## 1. categorize_math_tasks.py

### Beschreibung

Kategorisiert Mathematikaufgaben automatisch in fachliche Teilbereiche. Nutzt GPT-4o Vision, um das Aufgabenbild zu analysieren und eine mathematische Kategorie zuzuordnen.

### Kategorien

Das Skript verwendet fünf vordefinierte Kategorien:

| Kategorie | Beschreibung | Beispiele |
|-----------|--------------|-----------|
| **Arithmetik** | Rechnen mit Zahlen, Grundrechenarten | Addition, Subtraktion, Multiplikation, Division, Prozentrechnung |
| **Stochastik** | Wahrscheinlichkeitsrechnung und Statistik | Würfelexperimente, Kombinatorik, Datenauswertung |
| **Geometrie** | Formen, Flächen, Körper, räumliches Denken | Dreiecke, Kreise, Volumenberechnung, Symmetrie |
| **Algebra** | Gleichungen, Terme, Variablen | Gleichungssysteme, Funktionen, Term-Vereinfachung |
| **unknown** | Nicht eindeutig zuordenbar oder Fehler | Multi-Topic-Aufgaben, API-Fehler |

### Technische Details

**Modell**: GPT-4o (Vision-fähig)
- `max_tokens`: 50
- `temperature`: 0 (deterministisch)

**System Prompt**:
```
Du bist ein Experte für Mathematik-Klassifikation. 
Analysiere das Bild einer Mathematikaufgabe und ordne sie EINER dieser Kategorien zu:
- Arithmetik: Rechnen mit Zahlen (Addition, Subtraktion, Multiplikation, Division, Prozente)
- Stochastik: Wahrscheinlichkeitsrechnung, Kombinatorik, Statistik
- Geometrie: Formen, Flächen, Körper, räumliche Probleme
- Algebra: Gleichungen, Terme, Variablen, Funktionen

Antworte mit GENAU EINEM Wort: Arithmetik, Stochastik, Geometrie oder Algebra
```

**Workflow**:
1. Dataset (`dataset_final.json`) laden
2. Für jedes Bild ohne oder mit `math_category: "unknown"`:
   - Bild als Base64 encodieren
   - An GPT-4o Vision API senden mit System Prompt
   - Antwort validieren gegen Kategorien-Liste
   - Bei ungültiger Antwort: `"unknown"` setzen
3. Kategorie in Dataset speichern
4. Alle 10 Bilder Zwischenspeicherung

**Fehlerbehandlung**:
- Ungültige API-Antworten → `"unknown"`
- Bereits kategorisierte Bilder (außer `"unknown"`) werden übersprungen
- Fehlende Bilder → Warnung, fortfahren

**Performance**:
- Verarbeitet nur unkategorisierte oder `"unknown"` Bilder
- Kosten: ~$0.003 pro Bild (GPT-4o Vision)
- Geschwindigkeit: ~2-3 Sekunden pro Bild

### Output-Format

Fügt zu jedem Dataset-Eintrag hinzu:
```json
{
  "image_path": "dataset_final/1998_3und4_1.png",
  "math_category": "Arithmetik",
  ...
}
```

### Verwendung

```bash
cd /Users/dennishug/Desktop/vlm_repo
python src/categorize_math_tasks.py
```

**Statistik-Ausgabe**:
```
📊 Statistiken:
  Algebra        :  845 (23.8%)
  Arithmetik     : 1234 (34.7%)
  Geometrie      :  892 (25.1%)
  Stochastik     :  456 (12.8%)
  unknown        :  130 ( 3.6%)
```

---

## 2. extract_text.py

### Beschreibung

Extrahiert den vollständigen Text aus Aufgabenbildern mittels OCR. Erfasst sowohl die Fragestellung als auch alle Antwortoptionen (A-E) in strukturierter Form.

### Technische Details

**Modell**: GPT-4o-mini (kosteneffizienter als GPT-4o)
- `max_tokens`: 1000
- `temperature`: 0 (deterministisch)
- `response_format`: `{"type": "json_object"}` (erzwingt valides JSON)

**System Prompt**:
```
Du bist ein Experte für OCR und Textextraktion aus Mathematikaufgaben.

Extrahiere aus dem Bild:
1. **question**: Die vollständige Fragestellung (inkl. aller Kontext-Informationen)
2. **answer_options**: Liste aller Antwortmöglichkeiten (normalerweise A-E)

Wichtig:
- Extrahiere den Text genau wie er ist (inklusive mathematischer Notation)
- Behalte die Struktur und Formatierung bei
- Wenn keine expliziten Antwortoptionen vorhanden sind, setze answer_options auf leere Liste
- Gib NUR ein valides JSON zurück, keine zusätzlichen Erklärungen

Format:
{
  "question": "Die Fragestellung mit allen Details...",
  "answer_options": ["A) Antwort 1", "B) Antwort 2", "C) Antwort 3", "D) Antwort 4", "E) Antwort 5"]
}
```

**Workflow**:
1. Dataset laden
2. Extraktions-Cache laden (`data/text_extraction_cache.json`)
3. Für jedes Bild:
   - Prüfen ob bereits im Cache → überspringen
   - Bild als Base64 encodieren
   - An GPT-4o-mini senden mit System Prompt
   - JSON-Antwort parsen und validieren
   - In Dataset und Cache speichern
   - **5 Sekunden Pause** zwischen Anfragen (Rate Limiting)
4. Alle 10 Bilder: Dataset und Cache speichern

**Caching-System**:
- Verhindert doppelte API-Aufrufe für bereits extrahierte Bilder
- Cache-Datei: `data/text_extraction_cache.json`
- Format: `{"image_path": {"question": "...", "answer_options": [...]}}`
- Bei erneutem Script-Aufruf: Nur neue/fehlgeschlagene Bilder werden verarbeitet

**Fehlerbehandlung**:
- JSON Parse-Fehler → `None`, Warnung, fortfahren
- Fehlende Bilder → Warnung, fortfahren
- API-Fehler → `None`, Fehler-Log, fortfahren
- Unvollständige Daten (fehlende Keys) → Warnung, leere Werte einsetzen

**Performance**:
- Kosten: ~$0.0005 pro Bild (GPT-4o-mini)
- Geschwindigkeit: ~5 Sekunden pro Bild (inkl. Pause)
- Cache beschleunigt Wiederholungen massiv

### Output-Format

Fügt zu jedem Dataset-Eintrag hinzu:
```json
{
  "image_path": "dataset_final/2012_5und6_A1.png",
  "extracted_text": {
    "question": "Welches der folgenden Längenmaße ist das größte?",
    "answer_options": [
      "A) 2012 mm",
      "B) 201 cm",
      "C) 20 dm",
      "D) 2,5 m",
      "E) 0,002 km"
    ]
  },
  ...
}
```

### Verwendung

```bash
cd /Users/dennishug/Desktop/vlm_repo
python src/extract_text.py
```

**Statistik-Ausgabe**:
```
✅ Fertig!
   Neu extrahiert: 250
   Aus Cache: 3307
   Fehlgeschlagen: 0

📊 Statistiken:
  Neu extrahiert:     250 ( 7.0%)
  Aus Cache geladen: 3307 (93.0%)
  Fehlgeschlagen:       0 ( 0.0%)
```

---

## 3. analyze_text_only.py

### Beschreibung

Analysiert, ob eine Mathematikaufgabe **nur mit dem Text** oder ob sie **visuelle Elemente** (Diagramme, Geometrie-Figuren, Grafiken) zur Lösung benötigt. Dies ist entscheidend für die Evaluation von VLMs, da es text-basiertes Leseverständnis von echtem visuellem Verstehen unterscheidet.

### Kategorisierungs-Logik

**`is_text_only: true`** - Aufgabe ist rein textbasiert:
- Der Text allein enthält alle notwendigen Informationen
- Visuelle Elemente sind optional, dekorativ oder illustrativ
- Beispiele: Zahlenrätsel, Wortprobleme, logische Puzzles

**`is_text_only: false`** - Visuelle Elemente sind notwendig:
- Geometrische Formen müssen analysiert werden
- Diagramme/Grafiken/Tabellen enthalten Schlüsselinformationen
- Räumliche Anordnung oder visuelle Muster sind relevant
- Bilder zeigen Daten, die nicht im Text stehen
- Beispiele: Geometrieaufgaben, Diagrammanalyse, Musterfortführung

### Technische Details

**Modell**: GPT-4o (besseres visuelles Verständnis als gpt-4o-mini)
- `max_tokens`: 10
- `temperature`: 0 (deterministisch)

**System Prompt**:
```
Du bist ein Experte für Mathematikaufgaben-Analyse.
Analysiere das Bild einer Mathematikaufgabe und entscheide:

**is_text_only = true**: Wenn die Aufgabe NUR mit dem Text gelöst werden kann.
- Der Text allein enthält alle notwendigen Informationen
- Visuelle Elemente sind optional, dekorativ oder illustrativ
- Beispiele: Reine Textaufgaben, Zahlenrätsel, Wortprobleme

**is_text_only = false**: Wenn visuelle Elemente NOTWENDIG zur Lösung sind.
- Geometrische Formen müssen analysiert werden
- Diagramme, Grafiken oder Tabellen enthalten wichtige Informationen
- Räumliche Anordnung oder visuelle Muster sind relevant
- Bilder zeigen Daten, die nicht im Text stehen
- Beispiele: Geometrieaufgaben mit Figuren, Diagrammanalyse, Musterfortführung

Antworte NUR mit: true oder false
```

**Workflow**:
1. Dataset laden
2. Analyse-Cache laden (`data/text_only_analysis_cache.json`)
3. Für jedes Bild:
   - Prüfen ob bereits im Cache → überspringen
   - Bild als Base64 encodieren
   - An GPT-4o Vision API senden
   - Antwort parsen: `"true"` → `true`, `"false"` → `false`
   - Bei ungültiger Antwort: `false` (konservativ: visuelle Elemente annehmen)
   - In Dataset und Cache speichern
4. Alle 10 Bilder: Dataset und Cache speichern

**Caching-System**:
- Verhindert doppelte Analysen
- Cache-Datei: `data/text_only_analysis_cache.json`
- Format: `{"image_path": true/false}`
- Beschleunigt Wiederholungen massiv

**Fehlerbehandlung**:
- API-Fehler → `false` (konservativ)
- Ungültige Antwort (nicht "true"/"false") → `false`, Warnung
- Fehlende Bilder → Warnung, fortfahren

**Performance**:
- Kosten: ~$0.003 pro Bild (GPT-4o Vision)
- Geschwindigkeit: ~2-3 Sekunden pro Bild
- Keine künstliche Pause (Model ist robust)

### Output-Format

Fügt zu jedem Dataset-Eintrag hinzu:
```json
{
  "image_path": "dataset_final/2015_7und8_C5.png",
  "is_text_only": false,
  ...
}
```

### Verwendung

```bash
cd /Users/dennishug/Desktop/vlm_repo
python src/analyze_text_only.py
```

**Statistik-Ausgabe**:
```
✅ Fertig! 150 Bilder neu analysiert, 3407 übersprungen

📊 Statistiken:
  Nur Text benötigt:          1245 (35.0%)
  Visuelle Elemente benötigt: 2312 (65.0%)
  Bereits analysiert:         3407
```

---

## Vergleich der drei Skripte

| Aspekt | categorize_math_tasks | extract_text | analyze_text_only |
|--------|----------------------|--------------|-------------------|
| **Zweck** | Fachbereichs-Klassifikation | OCR von Frage + Antworten | Text-only vs. visuell |
| **Modell** | GPT-4o | GPT-4o-mini | GPT-4o |
| **Output-Typ** | String (Kategorie) | Nested JSON | Boolean |
| **Output-Feld** | `math_category` | `extracted_text` | `is_text_only` |
| **Antwortformat** | Einwort-Antwort | JSON Object | true/false |
| **max_tokens** | 50 | 1000 | 10 |
| **Pause zwischen Calls** | Nein | 5 Sekunden | Nein |
| **Cache-Datei** | Nein | `text_extraction_cache.json` | `text_only_analysis_cache.json` |
| **Kosten pro Bild** | ~$0.003 | ~$0.0005 | ~$0.003 |
| **Geschwindigkeit** | ~2-3s | ~5s (mit Pause) | ~2-3s |
| **Skip-Logik** | Überspringt bereits kategorisierte (außer "unknown") | Cache-basiert | Cache-basiert |

### Gesamt-Pipeline-Kosten

Für 3.557 Bilder (komplettes Dataset):
- **categorize_math_tasks**: 3.557 × $0.003 ≈ **$10.67**
- **extract_text**: 3.557 × $0.0005 ≈ **$1.78**
- **analyze_text_only**: 3.557 × $0.003 ≈ **$10.67**

**Gesamt**: ~**$23.12** für vollständige semantische Anreicherung

---

## Best Practices

### 1. Reihenfolge der Ausführung

**Empfohlene Reihenfolge**:
```bash
# 1. Zuerst Kategorisierung (unabhängig)
python src/categorize_math_tasks.py

# 2. Dann Textextraktion (kann parallel zu 1 laufen)
python src/extract_text.py

# 3. Zuletzt Text-Only-Analyse (profitiert von extracted_text für Verständnis)
python src/analyze_text_only.py
```

**Parallelisierung möglich**: Alle drei Skripte können theoretisch parallel laufen, da sie verschiedene Felder im Dataset bearbeiten.

### 2. Cache-Management

**Cache-Dateien**:
- `data/text_extraction_cache.json` (extract_text)
- `data/text_only_analysis_cache.json` (analyze_text_only)

**Empfehlungen**:
- **Nie löschen** (spart API-Kosten)
- Bei Dataset-Änderungen: Nur Cache-Einträge für geänderte Bilder entfernen
- Caches regelmäßig in Git committen (sind klein: <1 MB)

### 3. Fehlerbehandlung

Alle Skripte:
- Speichern regelmäßig (alle 10 Bilder)
- Bei Unterbrechung: Einfach erneut starten → fährt an letzter Stelle fort
- Loggen Fehler, aber brechen nicht ab

**Bei Fehlern**:
1. Check OpenAI API Key: `echo $OPENAI_API_KEY`
2. Check Internet-Verbindung
3. Check Rate Limits (selten bei GPT-4o)
4. Skript erneut starten (Cache verhindert doppelte Arbeit)

### 4. Qualitätskontrolle

**Stichproben-Prüfung empfohlen**:
```python
import json

# Lade Dataset
with open('dataset_final.json') as f:
    data = json.load(f)

# Prüfe 10 zufällige Einträge
import random
sample = random.sample(data, 10)
for entry in sample:
    print(f"Image: {entry['image_path']}")
    print(f"  Kategorie: {entry.get('math_category')}")
    print(f"  Frage: {entry.get('extracted_text', {}).get('question', '')[:60]}...")
    print(f"  Text-only: {entry.get('is_text_only')}")
    print()
```

### 5. API-Kosten optimieren

**Strategien**:
- **extract_text**: Nutzt GPT-4o-mini (5x günstiger als GPT-4o)
- **Caching**: Cache-Dateien verhindern doppelte API-Calls
- **Iteratives Vorgehen**: Erst kleine Test-Batches, dann vollständiges Dataset
- **Skip-Logik**: Bereits verarbeitete Einträge werden automatisch übersprungen

---

## Datenqualität und Limitationen

### Stärken

✅ **Automatisiert**: Keine manuelle Annotation nötig
✅ **Konsistent**: Deterministische Outputs (`temperature=0`)
✅ **Schnell**: ~5-8 Sekunden pro Bild für alle drei Analysen
✅ **Wiederholbar**: Cache ermöglicht identische Ergebnisse bei Wiederholung
✅ **Robust**: Fehler-tolerant, bricht nicht ab

### Limitationen

⚠️ **Kategorisierungs-Fehler**: Bei Multi-Topic-Aufgaben (z.B. Geometrie + Algebra) wird nur eine Kategorie gewählt
⚠️ **OCR-Fehler**: Mathematische Notation (Brüche, Wurzeln, Symbole) kann fehlerhaft sein
⚠️ **Text-Only Ambiguität**: Grenzfälle (z.B. kleine dekorative Grafiken) sind subjektiv
⚠️ **Modell-Bias**: GPT-4o könnte systematische Fehler bei bestimmten Aufgaben-Typen haben
⚠️ **Kosten**: Bei großen Datasets (>10.000 Bilder) werden API-Kosten relevant

### Empfohlene Validierung

Für wissenschaftliche Arbeiten (z.B. Studienarbeit/Bachelor/Master):

1. **Stichproben-Validierung**: 50-100 zufällige Einträge manuell prüfen
2. **Inter-Rater-Reliability**: Zweite Person kategorisiert Stichprobe → Vergleich
3. **Fehleranalyse**: Systematische Fehler identifizieren (z.B. "Geometrie wird oft falsch klassifiziert")
4. **Dokumentation**: Limitationen im Methodenteil erwähnen

---

## Integration in Evaluations-Pipeline

### Verwendung der Metadaten

**1. Fachbereichs-spezifische Evaluation**:
```python
# Geometrie-Performance isoliert testen
geometrie_tasks = [t for t in dataset if t.get('math_category') == 'Geometrie']
evaluate_vlm(model, geometrie_tasks)
```

**2. Text-only vs. Visuell**:
```python
# VLM-Performance bei reinen Textaufgaben vs. visuellen Aufgaben
text_only = [t for t in dataset if t.get('is_text_only') == True]
visual = [t for t in dataset if t.get('is_text_only') == False]

text_acc = evaluate_vlm(model, text_only)
visual_acc = evaluate_vlm(model, visual)

print(f"Text-only: {text_acc:.1%}, Visual: {visual_acc:.1%}")
```

**3. Textuelle Analyse**:
```python
# Schwierigkeitsanalyse basierend auf Fragelänge
questions = [t['extracted_text']['question'] for t in dataset if 'extracted_text' in t]
avg_length = sum(len(q.split()) for q in questions) / len(questions)
```

### Dataset-Statistiken

**Nach vollständiger Anreicherung**:
```python
import json
from collections import Counter

with open('dataset_final.json') as f:
    data = json.load(f)

# Kategorie-Verteilung
categories = Counter(t.get('math_category') for t in data)
print("Kategorien:", dict(categories))

# Text-only Anteil
text_only_count = sum(1 for t in data if t.get('is_text_only') == True)
print(f"Text-only: {text_only_count}/{len(data)} ({text_only_count/len(data)*100:.1f}%)")

# Extraktions-Erfolgsrate
extracted = sum(1 for t in data if 'extracted_text' in t and t['extracted_text'])
print(f"Text extrahiert: {extracted}/{len(data)} ({extracted/len(data)*100:.1f}%)")
```

---

## Fazit

Die drei OpenAI-basierten Skripte bilden eine **semantische Anreicherungs-Pipeline**, die das Känguru-Dataset von reinen Aufgabenbildern zu einem **multi-dimensionalen Benchmark** transformiert:

- **Fachliche Differenzierung** (Kategorien)
- **Modalitäts-Analyse** (Text-only vs. Visuell)
- **Textuelle Basis** (OCR für NLP-Analysen)

Dies ermöglicht **tiefere Einblicke** in VLM-Fähigkeiten und **präzisere wissenschaftliche Aussagen** über Stärken/Schwächen verschiedener Modelle.

**Für Studienarbeiten**: Diese Pipeline demonstriert modernen **AI-gestützten Daten-Engineering**-Ansatz und ist methodisch solide dokumentierbar.
