#!/usr/bin/env python3
"""
Einfaches Skript zur Analyse von PNG-Bildern mit potentiellen Artefakten.

Da wir kein OCR verwenden können, analysieren wir:
1. Bildgrößen und Proportionen
2. Dateinamen-Muster (welche Aufgaben sind betroffen)
3. Empfehlungen für Cropping basierend auf typischen Mustern
"""

from pathlib import Path
from collections import defaultdict
from PIL import Image


def analyze_images_simple(directory: Path):
    """Analysiert Bilder in einem Verzeichnis."""
    if not directory.exists():
        print(f"⚠️  Verzeichnis nicht gefunden: {directory}")
        return []
    
    results = []
    png_files = list(directory.glob('*.png'))
    
    for png_file in png_files:
        try:
            img = Image.open(png_file)
            width, height = img.size
            
            # Parse Dateiname
            name = png_file.stem
            parts = name.split('_')
            
            results.append({
                'path': png_file,
                'name': name,
                'width': width,
                'height': height,
                'aspect_ratio': width / height if height > 0 else 0,
                'parts': parts
            })
        except Exception as e:
            print(f"⚠️  Fehler bei {png_file.name}: {e}")
    
    return results


def main():
    """Hauptfunktion."""
    print("\n" + "="*80)
    print("  🔍 BILD-ANALYSE: Dimensionen und Muster")
    print("="*80)
    
    dirs = [
        ('references', Path('data/references')),
        ('references_1998_2011', Path('data/references_1998_2011'))
    ]
    
    all_results = {}
    
    for name, directory in dirs:
        print(f"\n📁 Analysiere: {directory}")
        results = analyze_images_simple(directory)
        all_results[name] = results
        print(f"   Gefunden: {len(results)} Bilder")
    
    # Statistiken
    print(f"\n{'='*80}")
    print("📊 STATISTIKEN")
    print(f"{'='*80}")
    
    for dir_name, results in all_results.items():
        if not results:
            continue
        
        print(f"\n📁 {dir_name}:")
        print(f"   Anzahl: {len(results)}")
        
        heights = [r['height'] for r in results]
        widths = [r['width'] for r in results]
        
        print(f"   Höhe: min={min(heights)}, max={max(heights)}, avg={sum(heights)/len(heights):.0f}")
        print(f"   Breite: min={min(widths)}, max={max(widths)}, avg={sum(widths)/len(widths):.0f}")
        
        # Gruppiere nach Höhe (gerundet auf 50px)
        height_groups = defaultdict(list)
        for r in results:
            height_bucket = (r['height'] // 50) * 50
            height_groups[height_bucket].append(r)
        
        print(f"\n   Höhen-Verteilung (gruppiert):")
        for height_bucket in sorted(height_groups.keys()):
            count = len(height_groups[height_bucket])
            print(f"      {height_bucket}-{height_bucket+49} px: {count} Bilder")
        
        # Finde ungewöhnlich hohe Bilder (potentielle Kandidaten für Cropping)
        avg_height = sum(heights) / len(heights)
        tall_images = [r for r in results if r['height'] > avg_height * 1.3]
        
        if tall_images:
            print(f"\n   ⚠️  Ungewöhnlich hohe Bilder ({len(tall_images)}):")
            for r in tall_images[:10]:
                print(f"      {r['name']}: {r['width']}x{r['height']} px")
            if len(tall_images) > 10:
                print(f"      ... und {len(tall_images) - 10} weitere")
    
    # Spezielle Analyse: A8, B8, C8 Aufgaben (letzte Aufgabe einer Kategorie)
    print(f"\n{'='*80}")
    print("🎯 ANALYSE: Letzte Aufgaben pro Kategorie (A8, B8, C8, A10, B10, C10)")
    print(f"{'='*80}")
    
    for dir_name, results in all_results.items():
        if not results:
            continue
        
        # Finde Aufgaben die mit A8, B8, C8, A10, B10, C10 enden
        last_tasks = [r for r in results if any(r['name'].endswith(suffix) for suffix in ['_A8', '_B8', '_C8', '_A10', '_B10', '_C10'])]
        
        if last_tasks:
            print(f"\n📁 {dir_name}: {len(last_tasks)} letzte Aufgaben gefunden")
            
            # Zeige Beispiele
            for r in last_tasks[:15]:
                print(f"   {r['name']}: {r['width']}x{r['height']} px")
            if len(last_tasks) > 15:
                print(f"   ... und {len(last_tasks) - 15} weitere")
    
    print(f"\n{'='*80}")
    print("💡 EMPFEHLUNGEN")
    print(f"{'='*80}")
    print("""
Basierend auf der Analyse:

1. Prüfe manuell einige der "letzten Aufgaben" (A8, B8, C8, A10, B10, C10)
   - Diese Bilder könnten am Ende "X-Punkte-Aufgaben" Text enthalten
   
2. Ungewöhnlich hohe Bilder sind Kandidaten für Cropping

3. Empfohlener Cropping-Betrag: 60-80 px vom unteren Rand
   - Dies sollte den "Punkte-Aufgaben" Text entfernen
   - Ohne die eigentliche Aufgabe zu beschneiden

4. Erstelle Backup vor dem Cropping!

Nächste Schritte:
- Visuelle Inspektion einiger Beispielbilder
- Cropping-Skript mit konfigurierbarer Höhe erstellen
    """)
    
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
