"""
Skript zur Extraktion von Text und Antwortoptionen aus Mathematikaufgaben-Bildern.

Extrahiert für jede Aufgabe:
- question: Die Fragestellung
- answer_options: Liste der Antwortmöglichkeiten (A, B, C, D, E)

Speichert als nested JSON unter "extracted_text" in data/final/dataset.json.
Verwendet Cache um bereits extrahierte Bilder zu überspringen.
"""

import json
import os
import base64
import time
from pathlib import Path
from typing import Dict, List, Optional
from openai import OpenAI
from dotenv import load_dotenv

def encode_image(image_path: str) -> str:
    """Kodiert ein Bild als base64 String."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def extract_text(client: OpenAI, image_path: str) -> Optional[Dict[str, any]]:
    """
    Extrahiert Text und Antwortoptionen aus einem Aufgabenbild.
    
    Args:
        client: OpenAI client
        image_path: Pfad zum Bild
        
    Returns:
        Dictionary mit 'question' und 'answer_options' oder None bei Fehler
    """
    try:
        # Bild enkodieren
        base64_image = encode_image(image_path)
        
        # OpenAI Vision API aufrufen
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": """Du bist ein Experte für OCR und Textextraktion aus Mathematikaufgaben.

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
}"""
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Extrahiere die Frage und Antwortoptionen aus diesem Bild:"
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
            max_tokens=1000,
            temperature=0,
            response_format={"type": "json_object"}
        )
        
        # Antwort extrahieren und parsen
        content = response.choices[0].message.content.strip()
        extracted_data = json.loads(content)
        
        # Validierung
        if "question" not in extracted_data or "answer_options" not in extracted_data:
            print(f"⚠️  Unvollständige Daten für {image_path}")
            return {
                "question": extracted_data.get("question", ""),
                "answer_options": extracted_data.get("answer_options", [])
            }
        
        return extracted_data
        
    except json.JSONDecodeError as e:
        print(f"❌ JSON Parse-Fehler bei {image_path}: {str(e)}")
        return None
    except Exception as e:
        print(f"❌ Fehler bei {image_path}: {str(e)}")
        return None

def load_dataset(json_path: str) -> List[Dict]:
    """Lädt das Dataset aus der JSON-Datei."""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_dataset(json_path: str, data: List[Dict]):
    """Speichert das Dataset in die JSON-Datei."""
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_extraction_cache(cache_path: Path) -> Dict[str, Dict]:
    """Lädt den Cache der bereits extrahierten Texte."""
    if cache_path.exists():
        with open(cache_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def save_extraction_cache(cache_path: Path, cache: Dict[str, Dict]):
    """Speichert den Cache der extrahierten Texte."""
    with open(cache_path, 'w', encoding='utf-8') as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)

def main():
    """Hauptfunktion zur Textextraktion aller Bilder."""
    
    # Pfade
    base_dir = Path(__file__).parent.parent
    json_path = base_dir / "data/final/dataset.json"
    images_dir = base_dir / "data" / "dataset_final"
    cache_path = base_dir / "data" / "text_extraction_cache.json"
    
    # Lade .env Datei
    load_dotenv(base_dir / ".env")
    
    # OpenAI Client initialisieren
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("❌ OPENAI_API_KEY Umgebungsvariable nicht gesetzt!")
    
    client = OpenAI(api_key=api_key)
    
    # Dataset und Cache laden
    print("📂 Lade data/final/dataset.json...")
    dataset = load_dataset(str(json_path))
    print(f"✅ {len(dataset)} Einträge geladen")
    
    print("📂 Lade Extraktions-Cache...")
    extraction_cache = load_extraction_cache(cache_path)
    print(f"✅ {len(extraction_cache)} bereits extrahierte Bilder im Cache")
    
    # Zähler für Statistiken
    stats = {
        "extracted": 0,
        "from_cache": 0,
        "failed": 0
    }
    total = len(dataset)
    processed = 0
    
    # Durch alle Einträge iterieren
    for i, entry in enumerate(dataset, 1):
        image_path_rel = entry.get("image_path")
        
        # Prüfen ob bereits im Cache extrahiert
        if image_path_rel in extraction_cache:
            cached_data = extraction_cache[image_path_rel]
            entry["extracted_text"] = cached_data
            stats["from_cache"] += 1
            # print(f"⏭️  [{i}/{total}] {image_path_rel} aus Cache geladen")
            continue
        
        # Vollständigen Pfad erstellen
        image_path_full = base_dir / "data" / image_path_rel
        
        # Prüfen ob Bild existiert
        if not image_path_full.exists():
            print(f"⚠️  [{i}/{total}] Bild nicht gefunden: {image_path_rel}")
            stats["failed"] += 1
            processed += 1
            continue
        
        # Text extrahieren
        print(f"🔍 [{i}/{total}] Extrahiere Text aus {image_path_rel}...")
        extracted_data = extract_text(client, str(image_path_full))
        
        # 5 Sekunden Pause zwischen API-Anfragen
        time.sleep(5)
        
        if extracted_data is None:
            stats["failed"] += 1
            print(f"❌ [{i}/{total}] Extraktion fehlgeschlagen: {image_path_rel}")
            processed += 1
            continue
        
        # Daten im Dataset und Cache speichern
        entry["extracted_text"] = extracted_data
        extraction_cache[image_path_rel] = extracted_data
        stats["extracted"] += 1
        
        # Kurze Vorschau der Frage (erste 60 Zeichen)
        question_preview = extracted_data.get("question", "")[:60] + "..."
        num_options = len(extracted_data.get("answer_options", []))
        print(f"✅ [{i}/{total}] {image_path_rel}")
        print(f"   Frage: {question_preview}")
        print(f"   Optionen: {num_options}")
        
        processed += 1
        
        # Periodisch speichern (alle 10 Bilder)
        if processed % 10 == 0:
            save_dataset(str(json_path), dataset)
            save_extraction_cache(cache_path, extraction_cache)
            print(f"💾 Zwischenspeicherung nach {processed} Bildern")
    
    # Finales Speichern
    save_dataset(str(json_path), dataset)
    save_extraction_cache(cache_path, extraction_cache)
    print(f"\n✅ Fertig!")
    print(f"   Neu extrahiert: {stats['extracted']}")
    print(f"   Aus Cache: {stats['from_cache']}")
    print(f"   Fehlgeschlagen: {stats['failed']}")
    
    # Statistiken ausgeben
    print("\n📊 Statistiken:")
    extracted_pct = (stats['extracted'] / total * 100) if total > 0 else 0
    cached_pct = (stats['from_cache'] / total * 100) if total > 0 else 0
    failed_pct = (stats['failed'] / total * 100) if total > 0 else 0
    print(f"  Neu extrahiert:    {stats['extracted']:4d} ({extracted_pct:5.1f}%)")
    print(f"  Aus Cache geladen: {stats['from_cache']:4d} ({cached_pct:5.1f}%)")
    print(f"  Fehlgeschlagen:    {stats['failed']:4d} ({failed_pct:5.1f}%)")

if __name__ == "__main__":
    main()
