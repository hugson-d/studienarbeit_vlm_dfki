#!/usr/bin/env python3
"""
Cleanup-Skript: Entfernt Dopplungen zwischen references und references_1998_2011.

Dieses Skript:
1. Scannt beide Verzeichnisse nach PNG-Dateien
2. Identifiziert Duplikate basierend auf Jahr und Klasse
3. Entfernt Duplikate aus dem jeweils nicht zugehörigen Verzeichnis
   - 1998-2011: bleiben in references_1998_2011
   - 2012-2025: bleiben in references
"""

import re
from pathlib import Path
from collections import defaultdict


def parse_filename(filename: str) -> tuple:
    """
    Extrahiert Jahr und Klasse aus Dateinamen.
    Format: YYYY_KlasseXXX_TaskID.png
    """
    # Pattern für beide Formate
    # 1998-2011: 1999_11bis13_1.png
    # 2012-2025: 2012_3und4_A1.png
    match = re.match(r'(\d{4})_([^_]+)_(.+)\.png$', filename)
    if not match:
        return None
    
    year = int(match.group(1))
    class_part = match.group(2)
    task_id = match.group(3)
    
    return year, class_part, task_id


def scan_directory(directory: Path) -> dict:
    """Scannt ein Verzeichnis und gruppiert Dateien nach Jahr."""
    files_by_year = defaultdict(list)
    
    if not directory.exists():
        print(f"⚠ Verzeichnis existiert nicht: {directory}")
        return files_by_year
    
    for png_file in directory.glob('*.png'):
        parsed = parse_filename(png_file.name)
        if parsed:
            year, class_part, task_id = parsed
            files_by_year[year].append(png_file)
    
    return files_by_year


def find_duplicate_files(directory: Path) -> list:
    """
    Findet Duplikat-Dateien mit " 2", " 3" etc. im Namen.
    z.B. datei.png und datei 2.png
    """
    duplicates = []
    
    if not directory.exists():
        return duplicates
    
    # Finde alle Dateien mit " 2", " 3" etc.
    for png_file in directory.glob('*.png'):
        filename = png_file.name
        # Pattern: endet mit " N.png" wobei N eine Zahl ist
        if re.search(r' \d+\.png$', filename):
            duplicates.append(png_file)
    
    return duplicates


def main():
    """Hauptfunktion."""
    print("\n" + "="*70)
    print("  🧹 Cleanup: Entferne Dopplungen aus references-Verzeichnissen")
    print("="*70)
    
    # Pfade definieren
    base_dir = Path('data')
    refs_1998_2011 = base_dir / 'references_1998_2011'
    refs_2012_2025 = base_dir / 'references'
    
    print(f"\n📂 Scanne Verzeichnisse...")
    print(f"   • {refs_1998_2011}")
    print(f"   • {refs_2012_2025}")
    
    # Scanne beide Verzeichnisse
    files_1998_2011 = scan_directory(refs_1998_2011)
    files_2012_2025 = scan_directory(refs_2012_2025)
    
    total_1998 = sum(len(files) for files in files_1998_2011.values())
    total_2012 = sum(len(files) for files in files_2012_2025.values())
    
    print(f"\n📊 Gefundene Dateien:")
    print(f"   • references_1998_2011: {total_1998} Dateien")
    print(f"   • references: {total_2012} Dateien")
    
    # Finde Duplikate mit " 2", " 3" etc. im Namen
    print(f"\n🔍 Suche nach Duplikat-Dateien (mit ' 2', ' 3' etc.)...")
    
    duplicates_1998 = find_duplicate_files(refs_1998_2011)
    duplicates_2012 = find_duplicate_files(refs_2012_2025)
    
    print(f"\n📋 Gefundene Duplikate:")
    print(f"   • references_1998_2011: {len(duplicates_1998)} Duplikate")
    print(f"   • references: {len(duplicates_2012)} Duplikate")
    
    # Finde Fehlplatzierungen (Jahre im falschen Verzeichnis)
    print(f"\n🔍 Suche nach Fehlplatzierungen (falsche Jahre)...")
    
    # Dateien aus 1998-2011 die in references liegen
    misplaced_in_2012 = []
    for year in range(1998, 2012):
        if year in files_2012_2025:
            misplaced_in_2012.extend(files_2012_2025[year])
    
    # Dateien aus 2012-2025 die in references_1998_2011 liegen
    misplaced_in_1998 = []
    for year in range(2012, 2026):
        if year in files_1998_2011:
            misplaced_in_1998.extend(files_1998_2011[year])
    
    # Berichte Ergebnisse
    print(f"\n📋 Fehlplatzierte Dateien:")
    print(f"   • 1998-2011 Dateien in 'references': {len(misplaced_in_2012)}")
    print(f"   • 2012-2025 Dateien in 'references_1998_2011': {len(misplaced_in_1998)}")
    
    total_to_delete = len(duplicates_1998) + len(duplicates_2012) + len(misplaced_in_2012) + len(misplaced_in_1998)
    
    if total_to_delete == 0:
        print(f"\n✅ Keine Dopplungen gefunden! Alles ist korrekt sortiert.")
        return
    
    # Lösche Duplikate und fehlplatzierte Dateien
    total_deleted = 0
    
    # Lösche Duplikate mit " 2" etc.
    if duplicates_1998:
        print(f"\n🗑️  Entferne {len(duplicates_1998)} Duplikate aus 'references_1998_2011'...")
        for file_path in duplicates_1998:
            try:
                file_path.unlink()
                total_deleted += 1
                if len(duplicates_1998) <= 10:
                    print(f"   ✓ Gelöscht: {file_path.name}")
            except Exception as e:
                print(f"   ✗ Fehler beim Löschen von {file_path.name}: {e}")
        if len(duplicates_1998) > 10:
            print(f"   ✓ {total_deleted} Duplikate gelöscht")
    
    if duplicates_2012:
        print(f"\n🗑️  Entferne {len(duplicates_2012)} Duplikate aus 'references'...")
        deleted_count = 0
        for file_path in duplicates_2012:
            try:
                file_path.unlink()
                total_deleted += 1
                deleted_count += 1
                if len(duplicates_2012) <= 10:
                    print(f"   ✓ Gelöscht: {file_path.name}")
            except Exception as e:
                print(f"   ✗ Fehler beim Löschen von {file_path.name}: {e}")
        if len(duplicates_2012) > 10:
            print(f"   ✓ {deleted_count} Duplikate gelöscht")
    
    # Lösche fehlplatzierte Dateien
    if misplaced_in_2012:
        print(f"\n🗑️  Entferne {len(misplaced_in_2012)} Dateien aus 'references'...")
        for file_path in misplaced_in_2012:
            try:
                file_path.unlink()
                total_deleted += 1
                if len(misplaced_in_2012) <= 10:  # Zeige Details nur bei wenigen Dateien
                    print(f"   ✓ Gelöscht: {file_path.name}")
            except Exception as e:
                print(f"   ✗ Fehler beim Löschen von {file_path.name}: {e}")
    
    if misplaced_in_1998:
        print(f"\n🗑️  Entferne {len(misplaced_in_1998)} Dateien aus 'references_1998_2011'...")
        for file_path in misplaced_in_1998:
            try:
                file_path.unlink()
                total_deleted += 1
                if len(misplaced_in_1998) <= 10:  # Zeige Details nur bei wenigen Dateien
                    print(f"   ✓ Gelöscht: {file_path.name}")
            except Exception as e:
                print(f"   ✗ Fehler beim Löschen von {file_path.name}: {e}")
    
    print(f"\n✅ Cleanup abgeschlossen!")
    print(f"   Insgesamt {total_deleted} Dateien gelöscht.")
    
    # Zeige finale Statistik
    print(f"\n📊 Finale Verteilung:")
    
    # Neu scannen für finale Zahlen
    final_1998_2011 = scan_directory(refs_1998_2011)
    final_2012_2025 = scan_directory(refs_2012_2025)
    
    total_final_1998 = sum(len(files) for files in final_1998_2011.values())
    total_final_2012 = sum(len(files) for files in final_2012_2025.values())
    
    print(f"   • references_1998_2011:")
    print(f"     - Dateien: {total_final_1998}")
    if final_1998_2011:
        years = sorted(final_1998_2011.keys())
        print(f"     - Jahre: {min(years)}-{max(years)}")
    
    print(f"   • references:")
    print(f"     - Dateien: {total_final_2012}")
    if final_2012_2025:
        years = sorted(final_2012_2025.keys())
        print(f"     - Jahre: {min(years)}-{max(years)}")
    
    print()


if __name__ == '__main__':
    main()
