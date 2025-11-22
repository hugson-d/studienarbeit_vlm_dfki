"""
Skript zur Kategorisierung von Mathematikaufgaben-Bildern mit OpenAI Vision API.

Analysiert jedes Bild im dataset_final Ordner und bestimmt die Mathematik-Kategorie:
- Arithmetik
- Stochastik
- Geometrie
- Algebra
- unknown (wenn unsicher)

Die Ergebnisse werden in dataset_final.json gespeichert.
"""

import json
import os
import base64
from pathlib import Path
from typing import Dict, List
from openai import OpenAI
from dotenv import load_dotenv

# Kategorien
CATEGORIES = ["Arithmetik", "Stochastik", "Geometrie", "Algebra", "unknown"]

def encode_image(image_path: str) -> str:
    """Kodiert ein Bild als base64 String."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def categorize_image(client: OpenAI, image_path: str) -> str:
    """
    Analysiert ein Bild mit OpenAI Vision und bestimmt die Mathematik-Kategorie.
    
    Args:
        client: OpenAI client
        image_path: Pfad zum Bild
        
    Returns:
        Kategorie als String (Arithmetik, Stochastik, Geometrie, Algebra oder unknown)
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
                    "content": """Du bist ein Experte für Mathematik-Kategorisierung. 
Analysiere das Bild einer Mathematikaufgabe und bestimme die Kategorie.

Kategorien:
- Arithmetik: Rechnen mit Zahlen, Addition, Subtraktion, Multiplikation, Division, Bruchrechnung, Prozentrechnung
- Stochastik: Wahrscheinlichkeit, Statistik, Kombinatorik, Datenanalyse
- Geometrie: Formen, Flächen, Volumen, Winkel, räumliches Denken
- Algebra: Gleichungen, Funktionen, Terme, Variablen
- unknown: Wenn du unsicher bist oder die Aufgabe mehrere Kategorien umfasst

Antworte NUR mit einem der folgenden Wörter: Arithmetik, Stochastik, Geometrie, Algebra, unknown"""
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Welcher Mathematik-Kategorie gehört diese Aufgabe an?"
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
            max_tokens=50,
            temperature=0
        )
        
        # Antwort extrahieren und validieren
        category = response.choices[0].message.content.strip()
        
        # Validierung: Stelle sicher, dass eine gültige Kategorie zurückgegeben wird
        if category not in CATEGORIES:
            print(f"⚠️  Ungültige Kategorie '{category}' für {image_path}, setze auf 'unknown'")
            return "unknown"
        
        return category
        
    except Exception as e:
        print(f"❌ Fehler bei {image_path}: {str(e)}")
        return "unknown"

def load_dataset(json_path: str) -> List[Dict]:
    """Lädt das Dataset aus der JSON-Datei."""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_dataset(json_path: str, data: List[Dict]):
    """Speichert das Dataset in die JSON-Datei."""
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def main():
    """Hauptfunktion zur Kategorisierung aller Bilder."""
    
    # Pfade
    base_dir = Path(__file__).parent.parent
    json_path = base_dir / "dataset_final.json"
    images_dir = base_dir / "data" / "dataset_final"
    
    # Lade .env Datei
    load_dotenv(base_dir / ".env")
    
    # OpenAI Client initialisieren
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("❌ OPENAI_API_KEY Umgebungsvariable nicht gesetzt!")
    
    client = OpenAI(api_key=api_key)
    
    # Dataset laden
    print("📂 Lade dataset_final.json...")
    dataset = load_dataset(str(json_path))
    print(f"✅ {len(dataset)} Einträge geladen")
    
    # Zähler für Statistiken
    stats = {cat: 0 for cat in CATEGORIES}
    total = len(dataset)
    processed = 0
    skipped = 0
    
    # Durch alle Einträge iterieren
    for i, entry in enumerate(dataset, 1):
        image_path_rel = entry.get("image_path")
        current_category = entry.get("math_category")
        
        # Wenn bereits kategorisiert (nicht None/null und nicht "unknown"), überspringen
        if current_category is not None and current_category != "unknown":
            print(f"⏭️  [{i}/{total}] {image_path_rel} bereits kategorisiert als '{current_category}'")
            stats[current_category] += 1
            skipped += 1
            continue
        
        # Vollständigen Pfad erstellen
        image_path_full = base_dir / "data" / image_path_rel
        
        # Prüfen ob Bild existiert
        if not image_path_full.exists():
            print(f"⚠️  [{i}/{total}] Bild nicht gefunden: {image_path_rel}")
            stats["unknown"] += 1
            processed += 1
            continue
        
        # Bild kategorisieren
        print(f"🔍 [{i}/{total}] Analysiere {image_path_rel}...")
        category = categorize_image(client, str(image_path_full))
        
        # Kategorie im Dataset aktualisieren
        entry["math_category"] = category
        stats[category] += 1
        processed += 1
        
        print(f"✅ [{i}/{total}] {image_path_rel} → {category}")
        
        # Periodisch speichern (alle 10 Bilder)
        if processed % 10 == 0:
            save_dataset(str(json_path), dataset)
            print(f"💾 Zwischenspeicherung nach {processed} Bildern")
    
    # Finales Speichern
    save_dataset(str(json_path), dataset)
    print(f"\n✅ Fertig! {processed} Bilder neu kategorisiert, {skipped} übersprungen")
    
    # Statistiken ausgeben
    print("\n📊 Statistiken:")
    for category, count in sorted(stats.items()):
        percentage = (count / total * 100) if total > 0 else 0
        print(f"  {category:15s}: {count:4d} ({percentage:5.1f}%)")

if __name__ == "__main__":
    main()
