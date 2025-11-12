"""
Test-Skript für Extraktion von Känguru-Aufgaben 1998-2011.

Diese älteren PDFs haben eine andere Struktur:
- Nummerierung: 1. 2. 3. 4. etc. (statt A1, B1, etc.)
- Aufgaben starten nach dem Marker "3-Punkte- Aufgaben" oder "3 Punkte-Aufgaben"
- Labels sind auf gleicher vertikaler X-Position ausgerichtet
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
import re

import fitz  # type: ignore[import]


def load_solutions(solutions_path: Path) -> Dict[tuple, str]:
    """Load solutions and create lookup dict: (year, class, task_id) -> answer."""
    with open(solutions_path, 'r', encoding='utf-8') as f:
        solutions = json.load(f)
    
    lookup = {}
    for entry in solutions:
        key = (entry['Jahr'], entry['Klasse'], entry['Aufgabe'])
        lookup[key] = entry['Lösung']
    
    return lookup


def parse_pdf_filename(filename: str) -> Optional[tuple]:
    """Extract year and class from filename like 'kaenguru2007_78.pdf'."""
    match = re.match(r'kaenguru(\d{4})_(\d+)\.pdf$', filename)
    if not match:
        return None
    
    year = int(match.group(1))
    class_code = match.group(2)
    
    # Map class code to standard format
    class_map = {
        '34': '3 und 4',
        '56': '5 und 6',
        '78': '7 und 8',
        '910': '9 und 10',
        '1113': '11 bis 13'
    }
    
    class_name = class_map.get(class_code)
    if not class_name:
        return None
    
    return year, class_name


def find_task_start_marker(doc):
    """Find the page and position where tasks start (after '3-Punkte- Aufgaben' or '3 Punkte-Aufgaben')."""
    markers = [
        '3-Punkte- Aufgaben',
        '3 Punkte-Aufgaben',
        '3-Punkte-Aufgaben',
        '3 Punkte Aufgaben',
        '3-Punkte-',
        '3 Punkte-',
        '3-Punkt-',
        '3 Punkt-'
    ]
    
    for page_idx, page in enumerate(doc):
        text = page.get_text('text')
        words = page.get_text('words')
        
        # Try direct text search first
        for marker in markers:
            if marker in text:
                # Find y-position after this marker
                # Look for the marker in words
                for i, (x0, y0, x1, y1, w, *_) in enumerate(words):
                    if marker[:5] in w:  # Look for start of marker (e.g., "3-Pun")
                        # Find where this line ends
                        line_y = y1
                        return page_idx, line_y
        
        # Alternative: Look for pattern "3" followed by "Punkte" within a few words
        for i, (x0, y0, x1, y1, w, *_) in enumerate(words):
            if w.strip() == '3' or w.strip() == '3-Punkte-' or w.strip() == '3-Punkt-':
                # Check next few words for "Punkte" or "Punkt"
                for j in range(i+1, min(i+5, len(words))):
                    next_word = words[j][4].lower()
                    if 'punkt' in next_word and 'aufgabe' in text[text.find(w):text.find(w)+100].lower():
                        return page_idx, words[j][3]  # y1 of "Punkte" word
    
    return None, None


def extract_items_from_pdf(pdf_path, year, class_code, solutions, output_dir):
    """Extract individual task items from old-format PDF (1998-2011)."""
    items = []
    doc = fitz.open(pdf_path)
    
    # Find where tasks start (after marker)
    start_page, start_y = find_task_start_marker(doc)
    if start_page is None:
        print(f"  ⚠ Could not find task start marker")
        doc.close()
        return items
    
    print(f"  Found task start at page {start_page + 1}, y={start_y:.1f}")
    
    # Expected task numbers for 1998-2011 format
    # Klasse 3-4: 21 tasks
    # Klasse 5-6, 7-8, 9-10, 11-13: 30 tasks each
    if class_code in ['3 und 4']:
        max_tasks = 21
    else:
        max_tasks = 30
    
    expected_labels = [str(i) for i in range(1, max_tasks + 1)]
    
    # Collect ALL number occurrences across all pages (starting from start_page)
    all_anchors = []
    for page_idx in range(start_page, len(doc)):
        page = doc[page_idx]
        words = page.get_text('words')
        
        for (x0, y0, x1, y1, w, *_) in words:
            # Check if word is a task number followed by period (e.g., "1.", "2.", etc.)
            w_stripped = w.strip()
            
            # Must end with a period and the part before period must be a task number
            if w_stripped.endswith('.'):
                w_clean = w_stripped.rstrip('.')
                if w_clean in expected_labels:
                    # Skip if on first page and above start_y
                    if page_idx == start_page and y0 < start_y:
                        continue
                        
                    all_anchors.append({
                        'label': w_clean,
                        'page_idx': page_idx,
                        'y_pos': y0,
                        'x_pos': x0,
                        'rect': fitz.Rect(x0, y0, x1, y1)
                    })
    
    if not all_anchors:
        print(f"  ⚠ No task numbers found after marker")
        doc.close()
        return items
    
    print(f"  Found {len(all_anchors)} potential task labels")
    
    # Find the most common X position (= left margin where task labels are)
    # Group X positions with tolerance
    x_positions = [a['x_pos'] for a in all_anchors]
    x_tolerance = 10.0
    x_clusters = {}
    
    for x in x_positions:
        found_cluster = False
        for cluster_x in list(x_clusters.keys()):
            if abs(x - cluster_x) <= x_tolerance:
                x_clusters[cluster_x] += 1
                found_cluster = True
                break
        if not found_cluster:
            x_clusters[x] = 1
    
    # Find the X position with most occurrences (= main task column)
    main_x = max(x_clusters.keys(), key=lambda x: x_clusters[x])
    print(f"  Main X position: {main_x:.1f} ({x_clusters[main_x]} occurrences)")
    
    # Filter anchors: keep only those near the main X position
    filtered_anchors = [
        a for a in all_anchors 
        if abs(a['x_pos'] - main_x) <= x_tolerance
    ]
    
    # Sort by document order
    filtered_anchors.sort(key=lambda a: (a['page_idx'], a['y_pos']))
    
    # Remove duplicate labels (keep first occurrence of each)
    seen_labels = set()
    selected_anchors = []
    for anchor in filtered_anchors:
        if anchor['label'] not in seen_labels:
            selected_anchors.append(anchor)
            seen_labels.add(anchor['label'])
    
    print(f"  Selected {len(selected_anchors)} unique task labels")
    
    # Extract items with margins
    TOP, BOTTOM, SIDE = 6.0, 4.0, 6.0
    
    for idx, anchor in enumerate(selected_anchors):
        page_idx = anchor['page_idx']
        page = doc[page_idx]
        rect = page.rect
        label = anchor['label']
        y0 = anchor['y_pos']
        
        # Determine vertical bounds
        y_top = max(rect.y0, y0 - TOP)
        
        # Find next anchor on same page for bottom boundary
        next_on_page = None
        for next_anchor in selected_anchors[idx + 1:]:
            if next_anchor['page_idx'] == page_idx:
                next_on_page = next_anchor
                break
        
        if next_on_page:
            y_bot = min(rect.y1, next_on_page['y_pos'] - BOTTOM)
        else:
            y_bot = rect.y1
        
        if y_bot <= y_top + 1.0:
            y_bot = min(rect.y1, y_top + 60)
        
        clip = fitz.Rect(rect.x0 + SIDE, y_top, rect.x1 - SIDE, y_bot)
        
        # Render image at 180 DPI
        zoom = 180 / 72
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, clip=clip, alpha=False)
        
        # Create output path
        item_filename = f"{year}_{class_code.replace(' ', '')}_{label}.png"
        item_path = output_dir / item_filename
        item_path.parent.mkdir(parents=True, exist_ok=True)
        item_path.write_bytes(pix.tobytes('png'))
        
        # Extract text
        text = page.get_text('text', clip=clip)
        
        # Get solution - for old format, task_id is just the number
        solution_key = (year, class_code, label)
        answer = solutions.get(solution_key, 'UNKNOWN')
        
        items.append({
            'year': year,
            'class': class_code,
            'task_id': label,
            'answer': answer,
            'page_index': page_idx + 1,
            'is_text_only': False,
            'image_path': str(item_path.relative_to(output_dir.parent)),
            'text': (text or '').strip()
        })
    
    doc.close()
    return items


def main():
    """Main extraction routine for 1998-2011 test."""
    
    # Setup paths
    project_root = Path(__file__).parent.parent
    data_root = project_root / 'data'
    solutions_path = project_root / 'lösungen.json'
    output_dir = project_root / 'data' / 'test_references_1998_2011'
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load solutions
    print(f"Loading solutions from {solutions_path}...")
    solutions = load_solutions(solutions_path)
    print(f"Loaded {len(solutions)} solutions")
    
    # Find all PDFs (1998-2011 only)
    pdf_files = []
    for class_folder in data_root.iterdir():
        if not class_folder.is_dir() or class_folder.name.startswith('test_') or class_folder.name == 'references':
            continue
        
        for pdf_file in sorted(class_folder.glob('kaenguru*.pdf')):
            pdf_info = parse_pdf_filename(pdf_file.name)
            if pdf_info:
                year, class_name = pdf_info
                # Only process 1998-2011
                if 1998 <= year <= 2011:
                    pdf_files.append((pdf_file, year, class_name))
    
    print(f"Found {len(pdf_files)} PDFs to process (1998-2011)\n")
    
    # Process all PDFs
    all_items = []
    for pdf_path, year, class_name in pdf_files:
        print(f"Processing {pdf_path.name} ({year}, {class_name})...")
        
        try:
            items = extract_items_from_pdf(
                pdf_path,
                year,
                class_name,
                solutions,
                output_dir
            )
            all_items.extend(items)
            print(f"  ✓ Extracted {len(items)} tasks\n")
        except Exception as e:
            print(f"  ✗ Error: {e}\n")
    
    # Write JSONL manifest
    manifest_path = output_dir / 'tasks_manifest_1998_2011_test.jsonl'
    with open(manifest_path, 'w', encoding='utf-8') as f:
        for item in all_items:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"\n✓ Done! Extracted {len(all_items)} tasks")
    print(f"  Images: {output_dir}/")
    print(f"  Manifest: {manifest_path}")


if __name__ == '__main__':
    main()
