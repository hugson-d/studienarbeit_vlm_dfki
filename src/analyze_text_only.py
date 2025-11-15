"""
Skript zur Analyse, ob Mathematikaufgaben nur mit Text oder mit visuellen Elementen lösbar sind.

Analysiert jedes Bild im dataset_final Ordner und bestimmt:
- is_text_only: true  -> Aufgabe ist nur mit dem Text lösbar (visuelle Elemente sind optional/dekorativ)
- is_text_only: false -> Aufgabe benötigt visuelle Elemente (Diagramme, Formen, Grafiken) zur Lösung

Die Ergebnisse werden in dataset_final.json gespeichert.
"""

import json
import os
import base64
from pathlib import Path
from typing import Dict, List
from openai import OpenAI
from dotenv import load_dotenv

def encode_image(image_path: str) -> str:
    """Kodiert ein Bild als base64 String."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def analyze_text_only(client: OpenAI, image_path: str) -> bool:
    """
    Analysiert ein Bild und bestimmt, ob die Aufgabe nur mit Text lösbar ist.
    
    Args:
        client: OpenAI client
        image_path: Pfad zum Bild
        
    Returns:
        True wenn nur Text benötigt wird, False wenn visuelle Elemente nötig sind
    """
    try:
        # Bild enkodieren
        base64_image = encode_image(image_path)
        
        # OpenAI Vision API aufrufen
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": """Du bist ein Experte für Mathematikaufgaben-Analyse.
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

Antworte NUR mit: true oder false"""
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Ist diese Aufgabe nur mit dem Text lösbar (true) oder werden visuelle Elemente benötigt (false)?"
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=10,
            temperature=0
        )
        
        # Antwort extrahieren und parsen
        answer = response.choices[0].message.content.strip().lower()
        
        # Validierung und Parsing
        if answer == "true":
            return True
        elif answer == "false":
            return False
        else:
            print(f"⚠️  Unerwartete Antwort '{answer}' für {image_path}, setze auf False (visuell)")
            return False
        
    except Exception as e:
        print(f"❌ Fehler bei {image_path}: {str(e)}")
        return False

def load_dataset(json_path: str) -> List[Dict]:
    """Lädt das Dataset aus der JSON-Datei."""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_dataset(json_path: str, data: List[Dict]):
    """Speichert das Dataset in die JSON-Datei."""
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_analyzed_cache(cache_path: Path) -> Dict[str, bool]:
    """Lädt den Cache der bereits analysierten Bilder."""
    if cache_path.exists():
        with open(cache_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def save_analyzed_cache(cache_path: Path, cache: Dict[str, bool]):
    """Speichert den Cache der analysierten Bilder."""
    with open(cache_path, 'w', encoding='utf-8') as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)

def main():
    """Hauptfunktion zur Analyse aller Bilder."""
    
    # Pfade
    base_dir = Path(__file__).parent.parent
    json_path = base_dir / "dataset_final.json"
    images_dir = base_dir / "data" / "dataset_final"
    cache_path = base_dir / "data" / "text_only_analysis_cache.json"
    
    # Lade .env Datei
    load_dotenv(base_dir / ".env")
    
    # OpenAI Client initialisieren
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("❌ OPENAI_API_KEY Umgebungsvariable nicht gesetzt!")
    
    client = OpenAI(api_key=api_key)
    
    # Dataset und Cache laden
    print("📂 Lade dataset_final.json...")
    dataset = load_dataset(str(json_path))
    print(f"✅ {len(dataset)} Einträge geladen")
    
    print("📂 Lade Analyse-Cache...")
    analyzed_cache = load_analyzed_cache(cache_path)
    print(f"✅ {len(analyzed_cache)} bereits analysierte Bilder im Cache")
    
    # Zähler für Statistiken
    stats = {
        "text_only": 0,
        "visual_required": 0,
        "already_analyzed": 0
    }
    total = len(dataset)
    processed = 0
    
    # Durch alle Einträge iterieren
    for i, entry in enumerate(dataset, 1):
        image_path_rel = entry.get("image_path")
        
        # Prüfen ob bereits im Cache analysiert
        if image_path_rel in analyzed_cache:
            cached_value = analyzed_cache[image_path_rel]
            entry["is_text_only"] = cached_value
            stats["already_analyzed"] += 1
            if cached_value:
                stats["text_only"] += 1
            else:
                stats["visual_required"] += 1
            # print(f"⏭️  [{i}/{total}] {image_path_rel} aus Cache geladen: {cached_value}")
            continue
        
        # Vollständigen Pfad erstellen
        image_path_full = base_dir / "data" / image_path_rel
        
        # Prüfen ob Bild existiert
        if not image_path_full.exists():
            print(f"⚠️  [{i}/{total}] Bild nicht gefunden: {image_path_rel}")
            processed += 1
            continue
        
        # Bild analysieren
        print(f"🔍 [{i}/{total}] Analysiere {image_path_rel}...")
        is_text_only = analyze_text_only(client, str(image_path_full))
        
        # Wert im Dataset und Cache aktualisieren
        entry["is_text_only"] = is_text_only
        analyzed_cache[image_path_rel] = is_text_only
        
        if is_text_only:
            stats["text_only"] += 1
            print(f"✅ [{i}/{total}] {image_path_rel} → nur Text benötigt")
        else:
            stats["visual_required"] += 1
            print(f"✅ [{i}/{total}] {image_path_rel} → visuelle Elemente benötigt")
        
        processed += 1
        
        # Periodisch speichern (alle 10 Bilder)
        if processed % 10 == 0:
            save_dataset(str(json_path), dataset)
            save_analyzed_cache(cache_path, analyzed_cache)
            print(f"💾 Zwischenspeicherung nach {processed} Bildern")
    
    # Finales Speichern
    save_dataset(str(json_path), dataset)
    save_analyzed_cache(cache_path, analyzed_cache)
    print(f"\n✅ Fertig! {processed} Bilder neu analysiert, {stats['already_analyzed']} übersprungen")
    
    # Statistiken ausgeben
    print("\n📊 Statistiken:")
    text_pct = (stats["text_only"] / total * 100) if total > 0 else 0
    visual_pct = (stats["visual_required"] / total * 100) if total > 0 else 0
    print(f"  Nur Text benötigt:          {stats['text_only']:4d} ({text_pct:5.1f}%)")
    print(f"  Visuelle Elemente benötigt: {stats['visual_required']:4d} ({visual_pct:5.1f}%)")
    print(f"  Bereits analysiert:         {stats['already_analyzed']:4d}")

if __name__ == "__main__":
    main()
