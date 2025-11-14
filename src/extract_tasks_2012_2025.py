"""
Extract individual Kangaroo competition tasks as PNG images with metadata.

This script processes all PDFs in the data/ folder structure and:
1. Extracts each task as a separate PNG image
2. Matches tasks with their correct answers from lösungen.json
3. Outputs a JSONL manifest with year, class, answer, image path, and extracted text

Based on the anchor-based segmentation approach with search_for().
"""

import csv
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
    """Extract year and class from filename like 'kaenguru2025_78.pdf'."""
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


def extract_items_from_pdf(pdf_path, year, class_code, solutions, output_dir):
    """Extract individual task items from a PDF file."""
    items = []
    doc = fitz.open(pdf_path)
    
    # Define expected sequence based on class level
    # Klasse 3-4 and 5-6: A1-A8, B1-B8, C1-C8 (24 tasks)
    # Klasse 7-8, 9-10, 11-13: A1-A10, B1-B10, C1-C10 (30 tasks)
    if class_code in ['3 und 4', '5 und 6']:
        max_num = 9  # 1-8
    else:
        max_num = 11  # 1-10
    
    expected_sequence = [f"{s}{i}" for s in ['A', 'B', 'C'] for i in range(1, max_num)]
    anchor_set = set(expected_sequence)    # Collect ALL anchor occurrences across all pages with their positions
    all_anchors = []
    for page_idx, page in enumerate(doc):
        words = page.get_text('words')
        for (x0, y0, x1, y1, w, *_) in words:
            if w in anchor_set:
                all_anchors.append({
                    'label': w,
                    'page_idx': page_idx,
                    'y_pos': y0,
                    'x_pos': x0,
                    'rect': fitz.Rect(x0, y0, x1, y1)
                })
    
    if not all_anchors:
        doc.close()
        return items
    
    # Find the most common X position (= left margin where task labels are)
    # Group X positions with small tolerance (±10 points)
    x_positions = [a['x_pos'] for a in all_anchors]
    x_tolerance = 10.0
    x_clusters = {}
    
    for x in x_positions:
        # Find existing cluster or create new one
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
    
    # Extract items with margins
    TOP, BOTTOM, SIDE = 6.0, 4.0, 6.0
    items = []
    
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
        
        # Get solution
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
    """Main extraction routine."""
    
    # Setup paths
    project_root = Path(__file__).parent.parent
    data_root = project_root / 'data'
    solutions_path = project_root / 'lösungen_2012_2025.json'
    output_dir = project_root / 'data' / 'references'
    
    # Load solutions
    print(f"Loading solutions from {solutions_path}...")
    solutions = load_solutions(solutions_path)
    print(f"Loaded {len(solutions)} solutions")
    
    # Find all PDFs
    pdf_files = []
    for class_folder in data_root.iterdir():
        if not class_folder.is_dir() or class_folder.name == 'references':
            continue
        
        for pdf_file in sorted(class_folder.glob('kaenguru*.pdf')):
            pdf_info = parse_pdf_filename(pdf_file.name)
            if pdf_info:
                year, class_name = pdf_info
                # Only process PDFs from 2012 onwards (older PDFs have different format)
                if year >= 2012:
                    pdf_files.append((pdf_file, year, class_name))
    
    print(f"Found {len(pdf_files)} PDFs to process (2012+)")
    
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
            print(f"  → Extracted {len(items)} tasks")
        except Exception as e:
            print(f"  ✗ Error: {e}")
    
    # Write JSONL manifest
    manifest_path = output_dir / 'tasks_manifest_2012_2025.jsonl'
    with open(manifest_path, 'w', encoding='utf-8') as f:
        for item in all_items:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    # Write CSV manifest
    csv_path = output_dir / 'tasks_manifest_2012_2025.csv'
    if all_items:
        fieldnames = ['year', 'class', 'task_id', 'answer', 'page_index', 'is_text_only', 'image_path', 'text']
        with open(csv_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_items)
    
    print(f"\n✓ Done! Extracted {len(all_items)} tasks")
    print(f"  Images: {output_dir}/")
    print(f"  Manifest (JSONL): {manifest_path}")
    print(f"  Manifest (CSV): {csv_path}")


if __name__ == '__main__':
    main()
