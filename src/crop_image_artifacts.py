#!/usr/bin/env python3
"""
Cropping-Skript: Entfernt "X-Punkte-Aufgaben" Text vom unteren Rand der Bilder.

Dieses Skript:
1. Scannt alle PNG-Dateien in references und references_1998_2011
2. Schneidet vom unteren Rand ab (Standard: 70px)
3. Erstellt Backups der Originalbilder
4. Speichert die beschnittenen Bilder
"""

from pathlib import Path
from PIL import Image
import shutil


def crop_image_bottom(image_path: Path, crop_height: int = 70, backup: bool = True):
    """
    Schneidet einen Bereich vom unteren Rand eines Bildes ab.
    
    Args:
        image_path: Pfad zum Bild
        crop_height: Höhe die vom unteren Rand abgeschnitten wird (in px)
        backup: Ob ein Backup erstellt werden soll
    
    Returns:
        True wenn erfolgreich, False bei Fehler
    """
    try:
        img = Image.open(image_path)
        width, height = img.size
        
        # Überspringe sehr kleine Bilder
        if height <= crop_height + 50:
            return False
        
        # Erstelle Backup wenn gewünscht
        if backup:
            backup_dir = image_path.parent / 'backups'
            backup_dir.mkdir(exist_ok=True)
            backup_path = backup_dir / image_path.name
            if not backup_path.exists():
                shutil.copy2(image_path, backup_path)
        
        # Schneide vom unteren Rand ab
        new_height = height - crop_height
        cropped = img.crop((0, 0, width, new_height))
        
        # Speichere das beschnittene Bild
        cropped.save(image_path)
        
        return True
    
    except Exception as e:
        print(f"   ✗ Fehler bei {image_path.name}: {e}")
        return False


def should_crop_image(image_path: Path, avg_height: float) -> bool:
    """
    Entscheidet ob ein Bild beschnitten werden soll.
    
    Kriterien:
    - Bild ist überdurchschnittlich hoch
    - ODER Bild ist eine "letzte Aufgabe" (A8, B8, C8, A10, B10, C10)
    """
    try:
        img = Image.open(image_path)
        width, height = img.size
        
        name = image_path.stem
        
        # Ist es eine "letzte Aufgabe"?
        is_last_task = any(name.endswith(suffix) for suffix in ['_A8', '_B8', '_C8', '_A10', '_B10', '_C10'])
        
        # Ist es überdurchschnittlich hoch?
        is_tall = height > avg_height * 1.2
        
        return is_last_task or is_tall
    
    except Exception:
        return False


def process_directory(directory: Path, crop_height: int = 70, dry_run: bool = False):
    """
    Verarbeitet alle Bilder in einem Verzeichnis.
    
    Args:
        directory: Verzeichnis mit PNG-Dateien
        crop_height: Höhe die abgeschnitten wird
        dry_run: Wenn True, wird nur simuliert ohne zu ändern
    """
    if not directory.exists():
        print(f"⚠️  Verzeichnis nicht gefunden: {directory}")
        return
    
    png_files = list(directory.glob('*.png'))
    
    if not png_files:
        print(f"   Keine PNG-Dateien gefunden")
        return
    
    # Berechne durchschnittliche Höhe
    heights = []
    for png_file in png_files:
        try:
            img = Image.open(png_file)
            heights.append(img.size[1])
        except:
            pass
    
    avg_height = sum(heights) / len(heights) if heights else 0
    
    print(f"   Durchschnittliche Höhe: {avg_height:.0f} px")
    print(f"   Crop-Höhe: {crop_height} px")
    
    # Filtere Bilder die beschnitten werden sollen
    to_crop = [f for f in png_files if should_crop_image(f, avg_height)]
    
    print(f"   Zu bearbeiten: {len(to_crop)}/{len(png_files)} Bilder")
    
    if dry_run:
        print(f"   🔍 DRY-RUN: Keine Änderungen werden vorgenommen")
        print(f"\n   Beispiele (erste 10):")
        for img_path in to_crop[:10]:
            img = Image.open(img_path)
            print(f"      {img_path.name}: {img.size[0]}x{img.size[1]} → {img.size[0]}x{img.size[1]-crop_height}")
        if len(to_crop) > 10:
            print(f"      ... und {len(to_crop) - 10} weitere")
        return
    
    # Verarbeite Bilder
    success_count = 0
    skipped_count = 0
    
    for idx, img_path in enumerate(to_crop, 1):
        if idx % 50 == 0:
            print(f"      ... {idx}/{len(to_crop)} bearbeitet", end='\r')
        
        if crop_image_bottom(img_path, crop_height, backup=True):
            success_count += 1
        else:
            skipped_count += 1
    
    print(f"      ... {len(to_crop)}/{len(to_crop)} bearbeitet ✓")
    print(f"   ✓ Erfolgreich: {success_count}")
    if skipped_count > 0:
        print(f"   ⊘ Übersprungen: {skipped_count} (zu klein)")


def main():
    """Hauptfunktion."""
    print("\n" + "="*80)
    print("  ✂️  CROPPING: Entferne 'X-Punkte-Aufgaben' vom unteren Rand")
    print("="*80)
    
    # Konfiguration
    CROP_HEIGHT = 70  # Pixel vom unteren Rand abschneiden
    DRY_RUN = False   # Setze auf True für Test-Durchlauf
    
    dirs = [
        ('references', Path('data/references')),
        ('references_1998_2011', Path('data/references_1998_2011'))
    ]
    
    if DRY_RUN:
        print("\n⚠️  DRY-RUN MODUS: Es werden keine Änderungen vorgenommen!")
    else:
        print("\n📦 Backups werden im 'backups' Unterverzeichnis erstellt")
    
    print(f"✂️  Crop-Höhe: {CROP_HEIGHT} px vom unteren Rand\n")
    
    total_processed = 0
    
    for name, directory in dirs:
        print(f"{'='*80}")
        print(f"📁 Verarbeite: {directory}")
        print(f"{'='*80}")
        
        process_directory(directory, CROP_HEIGHT, DRY_RUN)
        
        print()
    
    print(f"{'='*80}")
    print("✅ Cropping abgeschlossen")
    print(f"{'='*80}")
    
    if not DRY_RUN:
        print("\n💡 Hinweise:")
        print("   • Backups wurden im 'backups' Unterverzeichnis erstellt")
        print("   • Prüfe einige Bilder visuell um sicherzustellen, dass das Cropping korrekt ist")
        print("   • Falls zu viel oder zu wenig abgeschnitten wurde:")
        print("     1. Stelle Backups wieder her: cp backups/* ./")
        print("     2. Passe CROP_HEIGHT im Skript an")
        print("     3. Führe das Skript erneut aus")
    
    print()


if __name__ == '__main__':
    main()
