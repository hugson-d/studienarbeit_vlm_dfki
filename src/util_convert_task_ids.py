#!/usr/bin/env python3
"""
Utility script to convert numeric task_ids (1998-2011) to ABC format (A1, B11, C21, etc.)
based on class-specific difficulty mapping.
"""

import json
from pathlib import Path


def load_json(file_path: Path) -> dict | list:
    """Load JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(file_path: Path, data: dict | list, indent: int = 2):
    """Save JSON file with formatting."""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def convert_task_id_to_abc(task_num: int, class_level: str) -> str:
    """
    Convert numeric task_id to ABC format based on class level.
    
    Klassenstufen 3und4 und 5und6:
    - A1-A8 (Tasks 1-8)
    - B1-B8 (Tasks 9-16)
    - C1-C8 (Tasks 17-24)
    
    Klassenstufen 7und8, 9und10, 11bis13:
    - A1-A10 (Tasks 1-10)
    - B1-B10 (Tasks 11-20)
    - C1-C10 (Tasks 21-30)
    
    Args:
        task_num: Numeric task ID (1-30)
        class_level: Class level (3und4, 5und6, etc.)
    
    Returns:
        ABC format task_id (e.g., A2, B1, C5)
    """
    if class_level in ['3und4', '5und6']:
        if task_num <= 8:
            return f'A{task_num}'
        elif task_num <= 16:
            return f'B{task_num - 8}'  # B1-B8
        else:  # 17-24
            return f'C{task_num - 16}'  # C1-C8
    else:  # 7und8, 9und10, 11bis13
        if task_num <= 10:
            return f'A{task_num}'
        elif task_num <= 20:
            return f'B{task_num - 10}'  # B1-B10
        else:  # 21-30
            return f'C{task_num - 20}'  # C1-C10


def convert_dataset_task_ids(dataset_path: Path, output_path: Path = None, dry_run: bool = False) -> dict:
    """
    Convert all numeric task_ids in dataset to ABC format.
    
    Args:
        dataset_path: Path to dataset_final.json
        output_path: Path to save updated dataset (default: overwrites dataset_path)
        dry_run: If True, only show what would be changed without saving
    
    Returns:
        Dictionary with conversion statistics
    """
    print(f"\n🔄 Lade Dataset...\n")
    
    dataset = load_json(dataset_path)
    
    stats = {
        'total': len(dataset),
        'converted': 0,
        'skipped': 0,
        'by_year': {},
        'conversions': []
    }
    
    print("🔍 Konvertiere Task-IDs...\n")
    
    for entry in dataset:
        task_id = entry.get('task_id', '')
        year = entry.get('year')
        class_level = entry.get('class', '')
        
        # Check if task_id is numeric (1998-2011 format)
        if task_id and task_id.isdigit():
            task_num = int(task_id)
            new_task_id = convert_task_id_to_abc(task_num, class_level)
            
            # Update entry
            old_task_id = task_id
            entry['task_id'] = new_task_id
            
            # Statistics
            stats['converted'] += 1
            if year not in stats['by_year']:
                stats['by_year'][year] = 0
            stats['by_year'][year] += 1
            
            stats['conversions'].append({
                'year': year,
                'class': class_level,
                'old': old_task_id,
                'new': new_task_id,
                'image_path': entry.get('image_path', '')
            })
            
            if stats['converted'] <= 10:  # Show first 10 conversions
                print(f"✅ {year} {class_level}: {old_task_id} → {new_task_id}")
        else:
            stats['skipped'] += 1
    
    if stats['converted'] > 10:
        print(f"... und {stats['converted'] - 10} weitere\n")
    
    # Save results
    if not dry_run:
        output_file = output_path if output_path else dataset_path
        save_json(output_file, dataset)
        print(f"💾 Dataset gespeichert: {output_file}\n")
    else:
        print(f"🔍 Dry-Run Modus - Keine Änderungen gespeichert\n")
    
    # Print statistics
    print("="*60)
    print("📊 STATISTIK")
    print("="*60)
    print(f"Gesamt Einträge:     {stats['total']}")
    print(f"Konvertiert:         {stats['converted']}")
    print(f"Übersprungen (ABC):  {stats['skipped']}")
    
    if stats['by_year']:
        print(f"\n📅 Konvertierungen nach Jahr:")
        for year in sorted(stats['by_year'].keys()):
            print(f"  {year}: {stats['by_year'][year]}")
    
    print("="*60)
    
    return stats


def verify_conversion(dataset_path: Path):
    """Verify that all numeric task_ids have been converted."""
    print(f"\n🔍 Verifiziere Konvertierung...\n")
    
    dataset = load_json(dataset_path)
    
    numeric_ids = []
    abc_ids = []
    
    for entry in dataset:
        task_id = entry.get('task_id', '')
        year = entry.get('year')
        
        if task_id:
            if task_id.isdigit():
                numeric_ids.append({
                    'year': year,
                    'task_id': task_id,
                    'image_path': entry.get('image_path', '')
                })
            else:
                abc_ids.append(task_id)
    
    print("="*60)
    print("📊 VERIFIKATION")
    print("="*60)
    print(f"ABC-Format Task-IDs:      {len(abc_ids)}")
    print(f"Numerische Task-IDs:      {len(numeric_ids)}")
    
    if numeric_ids:
        print(f"\n⚠️  Noch {len(numeric_ids)} numerische IDs gefunden:")
        for item in numeric_ids[:5]:
            print(f"  - {item['year']}: {item['task_id']} ({item['image_path']})")
        if len(numeric_ids) > 5:
            print(f"  ... und {len(numeric_ids) - 5} weitere")
    else:
        print(f"\n✅ Alle Task-IDs im ABC-Format!")
    
    print("="*60)
    
    return len(numeric_ids) == 0


def main():
    """Main function."""
    project_root = Path(__file__).parent.parent
    
    dataset_path = project_root / 'dataset_final.json'
    
    print("\n" + "="*60)
    print("🔧 UTIL_CONVERT_TASK_IDS: Numerisch → ABC Format")
    print("="*60)
    
    # Check if file exists
    if not dataset_path.exists():
        print(f"❌ Fehler: {dataset_path} nicht gefunden!")
        return
    
    # Convert task IDs
    stats = convert_dataset_task_ids(
        dataset_path=dataset_path,
        dry_run=False  # Set to True to test without saving
    )
    
    # Verify conversion
    if stats['converted'] > 0:
        print("\n")
        verify_conversion(dataset_path)
    
    print("\n✅ Fertig!\n")


if __name__ == '__main__':
    main()
