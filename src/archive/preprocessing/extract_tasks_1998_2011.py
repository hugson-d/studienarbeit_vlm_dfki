"""
Extraktions-Skript für Känguru-Aufgaben 1998-2011.

Diese älteren PDFs haben eine andere Struktur:
- Nummerierung: 1. 2. 3. 4. etc. (sequentiell, nicht A1, B1, etc.)
- Aufgaben starten nach dem Marker "3-Punkte- Aufgaben" oder "3 Punkte-Aufgaben"
- Labels sind auf gleicher vertikaler X-Position ausgerichtet

Verwendet lösungen_1998_2011.json mit sequentieller Nummerierung 1-30.
"""

import json
import csv
from pathlib import Path
from typing import Dict, List, Optional
import re
import io

import fitz  # type: ignore[import]
import pytesseract  # type: ignore[import]
from PIL import Image


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
    """Find the page and position where tasks start using OCR."""
    markers = [
        '3-Punkte-Aufgaben',
        '3 Punkte-Aufgaben',
        '3-Punkte- Aufgaben',
        '3 Punkte Aufgaben',
    ]
    
    for page_idx, page in enumerate(doc):
        try:
            # Convert page to image
            zoom = 2  # Higher resolution for better OCR
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            
            # Convert to PIL Image
            img_data = pix.tobytes('png')
            img = Image.open(io.BytesIO(img_data))
            
            # Run OCR with layout information
            ocr_data = pytesseract.image_to_data(img, lang='deu', output_type=pytesseract.Output.DICT)
            
            # Search for marker in OCR results
            for i, text_ocr in enumerate(ocr_data['text']):
                if text_ocr:
                    # Check if this text or nearby texts form the marker
                    window = ' '.join([t for t in ocr_data['text'][i:i+5] if t])
                    for marker in markers:
                        if marker.replace('-', '').replace(' ', '') in window.replace('-', '').replace(' ', ''):
                            # Found marker! Convert from image to PDF coordinates
                            y_img = ocr_data['top'][i] + ocr_data['height'][i]
                            y_pdf = y_img / zoom
                            return page_idx, y_pdf
        except Exception as e:
            print(f"  ⚠ OCR failed for page {page_idx + 1}: {e}")
            continue
    
    return None, None


def extract_items_from_pdf(pdf_path, year, class_code, output_dir):
    """Extract individual task items from old-format PDF (1998-2011)."""
    count = 0
    doc = fitz.open(pdf_path)
    
    # Find where tasks start (after marker)
    start_page, start_y = find_task_start_marker(doc)
    if start_page is None:
        print(f"  ⚠ Could not find task start marker")
        doc.close()
        return count
    
    print(f"  Found task start at page {start_page + 1}, y={start_y:.1f}")
    
    # Expected task numbers for 1998-2011 format
    # Klasse 3-4: 21 tasks
    # Klasse 5-6, 7-8, 9-10, 11-13: 30 tasks each
    if class_code in ['3 und 4']:
        max_tasks = 21
    else:
        max_tasks = 30
    
    expected_labels = [str(i) for i in range(1, max_tasks + 1)]
    
    # Collect ALL number occurrences across all pages using OCR
    all_anchors = []
    for page_idx in range(start_page, len(doc)):
        page = doc[page_idx]
        page_start_y = start_y if page_idx == start_page else 0
        
        try:
            # Convert page to image
            zoom = 2
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img_data = pix.tobytes('png')
            img = Image.open(io.BytesIO(img_data))
            
            # Run OCR
            ocr_data = pytesseract.image_to_data(img, lang='deu', output_type=pytesseract.Output.DICT)
            
            # Find task numbers (pattern: digit followed by period)
            pattern = re.compile(r'^(\d+)\.$')
            for i, text_ocr in enumerate(ocr_data['text']):
                if text_ocr:
                    match = pattern.match(text_ocr.strip())
                    if match:
                        label = match.group(1)
                        if label in expected_labels:
                            # Convert from image to PDF coordinates
                            x_img = ocr_data['left'][i]
                            y_img = ocr_data['top'][i]
                            x_pdf = x_img / zoom
                            y_pdf = y_img / zoom
                            
                            if y_pdf >= page_start_y:
                                all_anchors.append({
                                    'label': label,
                                    'page_idx': page_idx,
                                    'y_pos': y_pdf,
                                    'x_pos': x_pdf,
                                })
        except Exception as e:
            print(f"  ⚠ OCR failed for page {page_idx + 1}: {e}")
    
    if not all_anchors:
        print(f"  ⚠ No task numbers found after marker")
        doc.close()
        return count
    
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
        count += 1
    
    doc.close()
    return count


def main():
    """Main function."""
    # Paths
    data_dir = Path('data')
    output_dir = Path('data/references_1998_2011')
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all PDF files for years 1998-2011
    all_pdfs = []
    for class_dir in data_dir.glob('Klasse*'):
        for pdf in class_dir.glob('*.pdf'):
            result = parse_pdf_filename(pdf.name)
            if result:
                year, class_name = result
                if 1998 <= year <= 2011:
                    all_pdfs.append((pdf, year, class_name))
    
    # Sort by year and class
    all_pdfs.sort(key=lambda x: (x[1], x[2]))
    
    print(f"\n📄 Found {len(all_pdfs)} PDFs for years 1998-2011")
    print(f"🎯 Starting extraction...\n")
    
    # Extract all items
    total_count = 0
    for pdf_path, year, class_name in all_pdfs:
        print(f"Processing: {pdf_path.name} ({year}, {class_name})")
        count = extract_items_from_pdf(pdf_path, year, class_name, output_dir)
        total_count += count
        print(f"  ✓ Extracted {count} tasks\n")
    
    print(f"\n✅ Extraction complete!")
    print(f"   Total items: {total_count}")
    print(f"   Output directory: {output_dir}")


if __name__ == '__main__':
    main()
