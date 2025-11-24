#!/usr/bin/env python3
"""
Creates a JSON dataset file for all tasks in dataset_final folder.
Combines information from both 1998-2011 and 2012-2025 solution files.
"""

from pathlib import Path
import json
import re

def load_solutions(solutions_path: Path) -> dict:
    """Load solutions from JSON file into a dictionary keyed by join_key."""
    with open(solutions_path, 'r', encoding='utf-8') as f:
        solutions = json.load(f)
    
    # Check if it's the new flat format (dict) or old array format
    if isinstance(solutions, dict):
        # New format: already a flat key-value dict like {"2012_11bis13_A1": "C"}
        return solutions
    
    # Old format: array of objects with join_key
    solutions_dict = {}
    for solution in solutions:
        join_key = solution.get('join_key')
        if join_key:
            solutions_dict[join_key] = solution
    
    return solutions_dict

def parse_filename(filename: str) -> dict | None:
    """
    Parse filename to extract components.
    Format examples: 
    - 2024_11bis13_A1.png (2012-2025)
    - 1999_11bis13_1.png (1998-2011)
    
    For 1998-2011 format, converts numeric task_id to ABC format based on year and class:
    - 1998-1999, Klasse 3und4: 15 tasks total (A1-A5, B1-B5, C1-C5)
    - 2000-2009, Klasse 3und4: 21 tasks total (A1-A7, B1-B7, C1-C7)
    - 2010-2011, Klasse 3und4: 24 tasks total (A1-A8, B1-B8, C1-C8)
    - 1998-2009, Klasse 5und6: 30 tasks total (A1-A10, B1-B10, C1-C10)
    - 2010-2011, Klasse 5und6: 24 tasks total (A1-A8, B1-B8, C1-C8)
    - All other classes (7und8, 9und10, 11bis13): 30 tasks total (A1-A10, B1-B10, C1-C10)
    """
    # Try 2012-2025 format first (ABC format)
    pattern_2012 = r'(\d{4})_([^_]+)_([ABC]\d+)\.png'
    match = re.match(pattern_2012, filename)
    
    if match:
        year, class_level, task_id = match.groups()
        return {
            'year': int(year),
            'class': class_level,
            'task_id': task_id,
            'period': '2012-2025'
        }
    
    # Try 1998-2011 format (sequential numbering)
    pattern_1998 = r'(\d{4})_([^_]+)_(\d+)\.png'
    match = re.match(pattern_1998, filename)
    
    if match:
        year_str, class_level, task_num_str = match.groups()
        year = int(year_str)
        task_num = int(task_num_str)
        
        # Determine tasks per difficulty level based on year and class
        if class_level == '3und4':
            if year in [1998, 1999]:
                tasks_per_level = 5  # 15 tasks total: A1-A5, B1-B5, C1-C5
            elif 2000 <= year <= 2009:
                tasks_per_level = 7  # 21 tasks total: A1-A7, B1-B7, C1-C7
            else:  # 2010, 2011
                tasks_per_level = 8  # 24 tasks total: A1-A8, B1-B8, C1-C8
        elif class_level == '5und6':
            if year <= 2009:
                tasks_per_level = 10  # 30 tasks for 1998-2009: A1-A10, B1-B10, C1-C10
            else:  # 2010, 2011
                tasks_per_level = 8  # 24 tasks for 2010-2011: A1-A8, B1-B8, C1-C8
        else:
            tasks_per_level = 10  # 30 tasks for 7und8, 9und10, 11bis13
        
        # Convert numeric task_id to ABC format
        if task_num <= tasks_per_level:
            task_id = f"A{task_num}"
        elif task_num <= 2 * tasks_per_level:
            task_id = f"B{task_num - tasks_per_level}"
        else:
            task_id = f"C{task_num - 2 * tasks_per_level}"
        
        return {
            'year': year,
            'class': class_level,
            'task_id': task_id,
            'period': '1998-2011'
        }
    
    return None

def create_join_key(year: int, class_level: str, task_id: str) -> str:
    """
    Create join_key in the format used by solution files.
    Format: "Jahr_Klassenstufesans_Aufgabe"
    Example: "2012_11bis13_A1" or "1999_11bis13_1"
    """
    # Normalize class level (remove spaces)
    class_normalized = class_level.replace(' ', '').replace('und', 'und')
    return f"{year}_{class_normalized}_{task_id}"

def main():
    project_root = Path(__file__).parent.parent
    dataset_final = project_root / 'data' / 'dataset_final'
    output_path = project_root / 'dataset_final.json'
    
    # Load existing dataset if it exists
    existing_entries = {}
    if output_path.exists():
        print(f"📖 Loading existing dataset from {output_path}...")
        with open(output_path, 'r', encoding='utf-8') as f:
            existing_data = json.load(f)
        
        # Create lookup by image_path
        for entry in existing_data:
            existing_entries[entry['image_path']] = entry
        
        print(f"   Found {len(existing_entries)} existing entries")
    else:
        print(f"📝 No existing dataset found, creating new one...")
    
    # Load only 2012-2025 solution file
    solutions_2012_2025 = load_solutions(project_root / 'data' / 'lösungen_2012_2025.json')
    
    print(f"📚 Loaded {len(solutions_2012_2025)} solutions from 2012-2025")
    
    # Process all PNG files in dataset_final
    dataset = []
    processed = 0
    skipped = 0
    new_entries = 0
    missing_solution = 0
    missing_keys = []
    
    for image_file in sorted(dataset_final.glob('*.png')):
        parsed = parse_filename(image_file.name)
        
        if not parsed:
            print(f"⚠️  Could not parse filename: {image_file.name}")
            continue
        
        # Build image path
        image_path = f'dataset_final/{image_file.name}'
        
        # Check if entry already exists
        if image_path in existing_entries:
            # Keep existing entry (preserves math_category, extracted_text, etc.)
            dataset.append(existing_entries[image_path])
            skipped += 1
            continue
        
        # Create join key for new entry
        join_key = create_join_key(parsed['year'], parsed['class'], parsed['task_id'])
        
        # Get solution from appropriate solutions file
        solution = solutions_2012_2025.get(join_key)
        
        # Get answer - handle both new flat format (string) and old format (dict)
        if solution:
            if isinstance(solution, str):
                # New format: solution is already the answer string
                answer = solution
            else:
                # Old format: solution is a dict with 'Lösung' key
                answer = solution.get('Lösung', '')
        else:
            print(f"⚠️  No solution found for {image_file.name} (join_key: {join_key})")
            answer = ''
            missing_solution += 1
            missing_keys.append({
                'filename': image_file.name,
                'join_key': join_key,
                'year': parsed['year'],
                'class': parsed['class'],
                'task_id': parsed['task_id']
            })
        
        # Build dataset entry for new image
        entry = {
            'image_path': image_path,
            'year': parsed['year'],
            'class': parsed['class'],
            'task_id': parsed['task_id'],
            'answer': answer,
            'math_category': None,
            'is_text_only': False,
            'extracted_text': None
        }
        
        dataset.append(entry)
        new_entries += 1
        processed += 1
    
    # Sort dataset by year, class, and task_id
    def sort_key(entry):
        year = entry['year']
        class_order = {'3und4': 0, '5und6': 1, '7und8': 2, '9und10': 3, '11bis13': 4}
        class_level = entry['class'].replace(' ', '')
        class_num = class_order.get(class_level, 99)
        
        # Extract numeric part from task_id for sorting
        task_id = entry['task_id']
        if task_id[0].isalpha():  # ABC format
            difficulty = {'A': 0, 'B': 1, 'C': 2}.get(task_id[0], 99)
            number = int(task_id[1:])
            return (year, class_num, difficulty, number)
        else:  # Sequential format
            return (year, class_num, 0, int(task_id))
    
    dataset.sort(key=sort_key)
    
    # Write to JSON file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ Dataset updated successfully")
    print(f"   Total entries: {len(dataset)}")
    print(f"   Existing entries (kept): {skipped}")
    print(f"   New entries added: {new_entries}")
    print(f"📁 Output: {output_path}")
    
    if missing_solution > 0:
        print(f"⚠️  {missing_solution} tasks without solution")
        
        # Write missing keys to markdown
        missing_md_path = project_root / 'FEHLENDE_LÖSUNGEN.md'
        with open(missing_md_path, 'w', encoding='utf-8') as f:
            f.write("# Fehlende Lösungen\n\n")
            f.write(f"Es fehlen {missing_solution} Lösungen für folgende Aufgaben:\n\n")
            f.write("| Dateiname | Join Key | Jahr | Klasse | Aufgabe |\n")
            f.write("|-----------|----------|------|--------|----------|\n")
            
            for item in missing_keys:
                f.write(f"| {item['filename']} | `{item['join_key']}` | {item['year']} | {item['class']} | {item['task_id']} |\n")
        
        print(f"📝 Missing keys written to: {missing_md_path}")
    
    # Verification step: check if all PNGs have entries
    print(f"\n🔍 Verification:")
    all_pngs = set(f.name for f in dataset_final.glob('*.png'))
    dataset_pngs = set(Path(entry['image_path']).name for entry in dataset)
    
    missing_in_dataset = all_pngs - dataset_pngs
    
    if missing_in_dataset:
        print(f"❌ {len(missing_in_dataset)} PNG files are missing in dataset:")
        for png in sorted(missing_in_dataset)[:10]:  # Show first 10
            print(f"   - {png}")
        if len(missing_in_dataset) > 10:
            print(f"   ... and {len(missing_in_dataset) - 10} more")
    else:
        print(f"✅ All {len(all_pngs)} PNG files have entries in the dataset")
    
    # Print sample entries
    print(f"\n📊 Sample entries:")
    for entry in dataset[:3]:
        print(f"  - {entry['year']} {entry['class']} {entry['task_id']}: Answer {entry['answer']}")

if __name__ == '__main__':
    main()
