# Mapping-Logik für Känguru-Aufgaben

## Dateinamensformate

### Format 1: ABC-Notation (2012-2025)
```
{Jahr}_{Klassenstufe}_{Schwierigkeitsgrad}{Nummer}.png
Beispiel: 2024_7und8_B15.png
```

- **Jahr**: 2012-2025
- **Klassenstufe**: 3und4, 5und6, 7und8, 9und10, 11bis13
- **Task-ID**: A1-A10, B11-B20, C21-C30
- **Schwierigkeitsgrad**: Direkt aus Task-ID (A/B/C-Präfix)

### Format 2: Numerische Notation (1998-2011)
```
{Jahr}_{Klassenstufe}_{Nummer}.png
Beispiel: 2010_5und6_12.png
```

- **Jahr**: 1998-2011
- **Klassenstufe**: 3und4, 5und6, 7und8, 9und10, 11bis13
- **Task-ID**: 1-30 (numerisch)
- **Schwierigkeitsgrad**: Berechnet basierend auf Klassenstufe und Aufgabennummer

## Schwierigkeitsgrad-Zuordnung (1998-2011)

Die Zuordnung der Schwierigkeitsgrade hängt von der **Klassenstufe** ab:

### Klassenstufen 3und4 und 5und6
- **A (Leicht)**: Aufgaben 1-8
- **B (Mittel)**: Aufgaben 9-16
- **C (Schwer)**: Aufgaben 17-24

**Beispiele:**
- `2010_3und4_5.png` → Schwierigkeit A
- `2010_5und6_12.png` → Schwierigkeit B
- `2011_3und4_20.png` → Schwierigkeit C

### Klassenstufen 7und8, 9und10 und 11bis13
- **A (Leicht)**: Aufgaben 1-10
- **B (Mittel)**: Aufgaben 11-20
- **C (Schwer)**: Aufgaben 21-30

**Beispiele:**
- `2010_7und8_7.png` → Schwierigkeit A
- `2010_9und10_15.png` → Schwierigkeit B
- `2011_11bis13_25.png` → Schwierigkeit C

## Lösungen-Mapping

### Struktur der Lösungsdateien

#### lösungen_1998_2011.json
```json
{
  "2010_3und4_1": "E",
  "2010_3und4_2": "C",
  "2010_5und6_12": "B"
}
```

- **Key-Format**: `{Jahr}_{Klassenstufe}_{Nummer}`
- **Value**: Antwortbuchstabe (A-E)

#### lösungen_2012_2025.json
```json
{
  "2024_3und4_A1": "D",
  "2024_7und8_B15": "C",
  "2024_11bis13_C30": "A"
}
```

- **Key-Format**: `{Jahr}_{Klassenstufe}_{TaskID}`
- **Value**: Antwortbuchstabe (A-E)

### Mapping zu dataset_final.json

Das Mapping erfolgt über den `image_path`:

```json
{
  "image_path": "dataset_final/2010_3und4_1.png",
  "year": 2010,
  "class": "3und4",
  "task_id": "1",
  "answer": "E"
}
```

**Mapping-Prozess:**
1. Extrahiere Key aus `image_path`: `dataset_final/2010_3und4_1.png` → `2010_3und4_1`
2. Suche Key in entsprechender Lösungsdatei
3. Fülle `answer`-Feld mit gefundenem Wert

## Implementierung

### Parse-Funktion (analyze_dataset_distribution.py)

```python
def parse_filename(filename):
    # Try ABC format first (2012-2025)
    pattern_abc = r'(\d{4})_([^_]+)_([ABC]\d+)\.png'
    match = re.match(pattern_abc, filename)
    
    if match:
        year, class_level, task_id = match.groups()
        difficulty = task_id[0]  # A, B, or C
        task_number = int(task_id[1:])
        return {
            'year': int(year),
            'class_level': class_level,
            'task_id': task_id,
            'difficulty': difficulty,
            'task_number': task_number
        }
    
    # Try numeric format (1998-2011)
    pattern_numeric = r'(\d{4})_([^_]+)_(\d+)\.png'
    match = re.match(pattern_numeric, filename)
    
    if match:
        year, class_level, task_id = match.groups()
        task_num = int(task_id)
        
        # Klassenabhängige Schwierigkeitsgrad-Zuordnung
        if class_level in ['3und4', '5und6']:
            if task_num <= 8:
                difficulty = 'A'
            elif task_num <= 16:
                difficulty = 'B'
            else:  # 17-24
                difficulty = 'C'
        else:  # 7und8, 9und10, 11bis13
            if task_num <= 10:
                difficulty = 'A'
            elif task_num <= 20:
                difficulty = 'B'
            else:  # 21-30
                difficulty = 'C'
        
        return {
            'year': int(year),
            'class_level': class_level,
            'task_id': task_id,
            'difficulty': difficulty,
            'task_number': task_num
        }
    
    return None
```

## Statistiken

Nach dem Mapping (Stand: November 2025):

- **Gesamt**: 2060 Aufgaben
- **Jahre**: 2010-2025 (16 Jahre)
- **Klassenstufen**: 5 (3und4, 5und6, 7und8, 9und10, 11bis13)
- **Schwierigkeitsgrade**: ~33% A, ~33% B, ~33% C (ausgeglichen)
- **Gemappte Lösungen**: 259 Einträge aus lösungen_1998_2011.json

### Verteilung 2010-2011
- **2010**: 125 Aufgaben (6.1%)
- **2011**: 134 Aufgaben (6.5%)
