#!/usr/bin/env python3
"""
Utility script to map solutions from lösungen_1998_2011.json to data/final/dataset.json.
Maps answer fields based on image_path matching.
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


def extract_key_from_path(image_path: str) -> str:
    """
    Extract mapping key from image_path.
    Example: 'dataset_final/2010_3und4_1.png' -> '2010_3und4_1'
    """
    filename = Path(image_path).stem  # Remove .png extension
    return filename


def map_solutions_to_dataset(
    dataset_path: Path,
    solutions_path: Path,
    output_path: Path = None,
    dry_run: bool = False
) -> dict:
    """
    Map solutions from lösungen_1998_2011.json to data/final/dataset.json.
    
    Args:
        dataset_path: Path to data/final/dataset.json
        solutions_path: Path to lösungen_1998_2011.json
        output_path: Path to save updated dataset (default: overwrites dataset_path)
        dry_run: If True, only show what would be changed without saving
    
    Returns:
        Dictionary with mapping statistics
    """
    print(f"\n🔄 Lade Daten...\n")
    
    # Load files
    dataset = load_json(dataset_path)
    solutions = load_json(solutions_path)
    
    print(f"📂 Dataset: {len(dataset)} Einträge")
    print(f"📂 Lösungen: {len(solutions)} Einträge\n")
    
    # Statistics
    stats = {
        'total_dataset': len(dataset),
        'total_solutions': len(solutions),
        'mapped': 0,
        'already_filled': 0,
        'not_found': 0,
        'updated': []
    }
    
    # Map solutions
    print("🔍 Mappe Lösungen...\n")
    
    for entry in dataset:
        image_path = entry.get('image_path', '')
        key = extract_key_from_path(image_path)
        
        # Check if key exists in solutions
        if key in solutions:
            current_answer = entry.get('answer')
            new_answer = solutions[key]
            
            # Check if answer field is empty or needs update
            if not current_answer or current_answer == "":
                entry['answer'] = new_answer
                stats['mapped'] += 1
                stats['updated'].append({
                    'key': key,
                    'image_path': image_path,
                    'answer': new_answer
                })
                print(f"✅ {key}: {new_answer}")
            else:
                stats['already_filled'] += 1
                if current_answer != new_answer:
                    print(f"⚠️  {key}: Bereits gefüllt ({current_answer}), würde auf {new_answer} ändern")
    
    # Check for solutions not found in dataset
    dataset_keys = {extract_key_from_path(entry.get('image_path', '')) for entry in dataset}
    for solution_key in solutions.keys():
        if solution_key not in dataset_keys:
            stats['not_found'] += 1
    
    # Save results
    if not dry_run:
        output_file = output_path if output_path else dataset_path
        save_json(output_file, dataset)
        print(f"\n💾 Dataset gespeichert: {output_file}\n")
    else:
        print(f"\n🔍 Dry-Run Modus - Keine Änderungen gespeichert\n")
    
    # Print statistics
    print("="*60)
    print("📊 STATISTIK")
    print("="*60)
    print(f"Dataset Einträge:          {stats['total_dataset']}")
    print(f"Lösungen verfügbar:        {stats['total_solutions']}")
    print(f"Neu gemappt:               {stats['mapped']}")
    print(f"Bereits gefüllt:           {stats['already_filled']}")
    print(f"Nicht im Dataset gefunden: {stats['not_found']}")
    print("="*60)
    
    return stats


def verify_mapping(dataset_path: Path, solutions_path: Path):
    """
    Verify that all solutions from lösungen_1998_2011.json are correctly mapped.
    """
    print(f"\n🔍 Verifiziere Mapping...\n")
    
    dataset = load_json(dataset_path)
    solutions = load_json(solutions_path)
    
    # Create mapping from dataset
    dataset_map = {}
    for entry in dataset:
        key = extract_key_from_path(entry.get('image_path', ''))
        dataset_map[key] = entry.get('answer', '')
    
    # Compare
    correct = 0
    missing = 0
    mismatch = 0
    
    for solution_key, solution_answer in solutions.items():
        if solution_key not in dataset_map:
            print(f"❌ Nicht im Dataset: {solution_key}")
            missing += 1
        elif dataset_map[solution_key] == solution_answer:
            correct += 1
        else:
            print(f"⚠️  Mismatch: {solution_key}")
            print(f"   Erwartet: {solution_answer}")
            print(f"   Gefunden: {dataset_map[solution_key]}")
            mismatch += 1
    
    print("\n" + "="*60)
    print("📊 VERIFIKATION")
    print("="*60)
    print(f"Korrekt gemappt:   {correct}")
    print(f"Fehler/Mismatch:   {mismatch}")
    print(f"Nicht gefunden:    {missing}")
    print(f"Erfolgsrate:       {correct / len(solutions) * 100:.1f}%")
    print("="*60)
    
    return correct == len(solutions)


def main():
    """Main function."""
    project_root = Path(__file__).parent.parent
    
    dataset_path = project_root / 'data/final/dataset.json'
    solutions_path = project_root / 'data' / 'lösungen_1998_2011.json'
    
    print("\n" + "="*60)
    print("🔧 UTIL_MAPPING: Lösungen 1998-2011 Mapper")
    print("="*60)
    
    # Check if files exist
    if not dataset_path.exists():
        print(f"❌ Fehler: {dataset_path} nicht gefunden!")
        return
    
    if not solutions_path.exists():
        print(f"❌ Fehler: {solutions_path} nicht gefunden!")
        return
    
    # Map solutions
    stats = map_solutions_to_dataset(
        dataset_path=dataset_path,
        solutions_path=solutions_path,
        dry_run=False  # Set to True to test without saving
    )
    
    # Verify mapping
    if stats['mapped'] > 0:
        print("\n")
        verify_mapping(dataset_path, solutions_path)
    
    print("\n✅ Fertig!\n")


if __name__ == '__main__':
    main()
