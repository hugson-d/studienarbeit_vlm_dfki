# Kategorisierung von Mathematikaufgaben

## Überblick

Das Skript `categorize_math_tasks.py` analysiert Bilder von Mathematikaufgaben mithilfe der OpenAI Vision API (GPT-4o-mini) und kategorisiert sie automatisch in eine von fünf Kategorien:

- **Arithmetik**: Rechnen mit Zahlen, Addition, Subtraktion, Multiplikation, Division, Bruchrechnung, Prozentrechnung
- **Stochastik**: Wahrscheinlichkeit, Statistik, Kombinatorik, Datenanalyse
- **Geometrie**: Formen, Flächen, Volumen, Winkel, räumliches Denken
- **Algebra**: Gleichungen, Funktionen, Terme, Variablen
- **unknown**: Wenn das Modell unsicher ist oder die Aufgabe mehrere Kategorien umfasst

## Voraussetzungen

1. **OpenAI API Key**: Du benötigst einen gültigen OpenAI API Key
2. **Python-Abhängigkeiten**: Die `openai` Bibliothek muss installiert sein

## Installation

```bash
# OpenAI-Bibliothek installieren
uv pip install openai
```

## Verwendung

### 1. OpenAI API Key setzen

Erstelle eine `.env` Datei im Projektverzeichnis:

```bash
# Kopiere die Beispieldatei
cp .env.example .env

# Bearbeite die .env Datei und füge deinen API Key ein
# .env
OPENAI_API_KEY=dein-echter-api-key-hier
```

**Hinweis**: Die `.env` Datei ist bereits im `.gitignore` und wird nicht ins Git-Repository hochgeladen.

### 2. Skript ausführen

```bash
python src/categorize_math_tasks.py
```

## Funktionsweise

1. Das Skript lädt `dataset_final.json`
2. Für jeden Eintrag mit `math_category: "unknown"`:
   - Lädt das Bild aus dem `data/dataset_final/` Ordner
   - Sendet das Bild an GPT-4o-mini zur Analyse
   - Erhält die Kategorie zurück
   - Aktualisiert den Eintrag in der JSON
3. Speichert die Ergebnisse periodisch (alle 10 Bilder) und am Ende
4. Gibt Statistiken über die Kategorieverteilung aus

## Features

- ✅ **Intelligente Kategorisierung**: Nutzt GPT-4o-mini für präzise Bildanalyse
- ✅ **Fortsetzbare Verarbeitung**: Überspringt bereits kategorisierte Einträge
- ✅ **Automatisches Speichern**: Periodische Zwischenspeicherung alle 10 Bilder
- ✅ **Fehlerbehandlung**: Robuste Fehlerbehandlung bei API-Fehlern
- ✅ **Fortschrittsanzeige**: Echtzeit-Updates während der Verarbeitung
- ✅ **Statistiken**: Zusammenfassung der Kategorieverteilung am Ende

## Beispiel-Output

```
📂 Lade dataset_final.json...
✅ 2785 Einträge geladen
🔍 [1/2785] Analysiere dataset_final/2010_3und4_1.png...
✅ [1/2785] dataset_final/2010_3und4_1.png → Arithmetik
🔍 [2/2785] Analysiere dataset_final/2010_3und4_2.png...
✅ [2/2785] dataset_final/2010_3und4_2.png → Geometrie
...
💾 Zwischenspeicherung nach 10 Bildern
...

✅ Fertig! 2785 Bilder neu kategorisiert, 0 übersprungen

📊 Statistiken:
  Algebra        :  456 ( 16.4%)
  Arithmetik     :  892 ( 32.0%)
  Geometrie      :  621 ( 22.3%)
  Stochastik     :  534 ( 19.2%)
  unknown        :  282 ( 10.1%)
```

## Kosten-Hinweis

Das Skript verwendet GPT-4o-mini, welches sehr kostengünstig ist:
- ~$0.00015 pro Bild (Eingabe) + ~$0.0003 pro Bild (Ausgabe)
- Für ~2800 Bilder: ca. $1.30 USD Gesamtkosten

## Anpassungen

Du kannst im Skript folgendes anpassen:

- **Modell**: Ändere `model="gpt-4o-mini"` zu einem anderen OpenAI Vision Modell
- **Kategorien**: Passe die `CATEGORIES` Liste und den System-Prompt an
- **Speicher-Frequenz**: Ändere `if processed % 10 == 0:` für häufigeres/selteneres Speichern
- **Temperature**: Ändere `temperature=0` für deterministischere (0) oder kreativere (höher) Antworten
