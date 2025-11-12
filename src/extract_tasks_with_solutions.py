"""
Extract individual Kangaroo competition tasks as PNG images with metadata.

This script processes all PDFs in the data/ folder structure and:
1. Extracts each task as a separate PNG image
2. Matches tasks with their correct answers from lösungen.json
3. Outputs a JSONL manifest with year, class, answer, image path, and extracted text

Based on the anchor-based segmentation approach with search_for().
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


def extract_items_from_pdf(
    pdf_path: Path,
    year: int,
    class_name: str,
    output_dir: Path,
    solutions: Dict[tuple, str],
    dpi: int = 180
) -> List[Dict]:
    """Extract all task items from a single PDF."""
    
    doc = fitz.open(pdf_path)
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    
    # Define all possible anchors
    anchors = [f"{s}{i}" for s in ['A', 'B', 'C'] for i in range(1, 11)]
    anchor_set = set(anchors)
    
    def find_anchors(page):
        """Find task label positions on a page."""
        words = page.get_text('words')
        res = []
        for (x0, y0, x1, y1, w, *_) in words:
            if w in anchor_set:
                res.append((w, y0, fitz.Rect(x0, y0, x1, y1)))
        res.sort(key=lambda t: (t[1], t[2].x0))
        return res
    
    # Collect anchors from all pages
    page_infos = []
    for i, page in enumerate(doc):
        page_infos.append({
            'index': i,
            'rect': page.rect,
            'anchors': find_anchors(page)
        })
    
    # Extract items with margins
    TOP, BOTTOM, SIDE = 6.0, 4.0, 6.0
    items = []
    
    for info in page_infos:
        i = info['index']
        page = doc[i]
        rect = page.rect
        chain = info['anchors']
        
        if not chain:
            continue
        
        for j, (label, y0, _) in enumerate(chain):
            y_top = max(rect.y0, y0 - TOP)
            y_next = chain[j + 1][1] if j + 1 < len(chain) else rect.y1
            y_bot = min(rect.y1, y_next - BOTTOM)
            
            if y_bot <= y_top + 1.0:
                y_bot = min(rect.y1, y_top + 60)
            
            clip = fitz.Rect(rect.x0 + SIDE, y_top, rect.x1 - SIDE, y_bot)
            
            # Render image
            pix = page.get_pixmap(matrix=mat, clip=clip, alpha=False)
            
            # Create output path
            item_filename = f"{year}_{class_name.replace(' ', '')}_{label}.png"
            item_path = output_dir / item_filename
            item_path.parent.mkdir(parents=True, exist_ok=True)
            item_path.write_bytes(pix.tobytes('png'))
            
            # Extract text
            text = page.get_text('text', clip=clip)
            
            # Get solution
            solution_key = (year, class_name, label)
            answer = solutions.get(solution_key, 'UNKNOWN')
            
            items.append({
                'year': year,
                'class': class_name,
                'task_id': label,
                'answer': answer,
                'page_index': i + 1,
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
    solutions_path = project_root / 'lösungen.json'
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
                pdf_files.append((pdf_file, year, class_name))
    
    print(f"Found {len(pdf_files)} PDFs to process")
    
    # Process all PDFs
    all_items = []
    for pdf_path, year, class_name in pdf_files:
        print(f"Processing {pdf_path.name} ({year}, {class_name})...")
        
        try:
            items = extract_items_from_pdf(
                pdf_path,
                year,
                class_name,
                output_dir,
                solutions,
                dpi=180
            )
            all_items.extend(items)
            print(f"  → Extracted {len(items)} tasks")
        except Exception as e:
            print(f"  ✗ Error: {e}")
    
    # Write JSONL manifest
    manifest_path = output_dir / 'tasks_manifest.jsonl'
    with open(manifest_path, 'w', encoding='utf-8') as f:
        for item in all_items:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"\n✓ Done! Extracted {len(all_items)} tasks")
    print(f"  Images: {output_dir}/")
    print(f"  Manifest: {manifest_path}")


if __name__ == '__main__':
    main()
