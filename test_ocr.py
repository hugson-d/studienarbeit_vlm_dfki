"""
OCR-based extraction for PDFs with encoding issues.
Converts pages to PNG, uses OCR to find task markers, applies old extraction logic.
"""

import sys
import fitz  # type: ignore[import]
from pathlib import Path
import re
import pytesseract  # type: ignore[import]
from PIL import Image
import io


def parse_pdf_filename(filename: str):
    """Extract year and class from filename."""
    match = re.match(r'kaenguru(\d{4})_(\d+)\.pdf$', filename)
    if not match:
        return None
    
    year = int(match.group(1))
    class_code = match.group(2)
    
    class_map = {
        '34': '3und4',
        '56': '5und6',
        '78': '7und8',
        '910': '9und10',
        '1113': '11bis13'
    }
    
    class_name = class_map.get(class_code)
    if not class_name:
        return None
    
    return year, class_name


def find_task_start_marker_ocr(doc):
    """Find the page and position where tasks start using OCR."""
    markers = [
        '3-Punkte-Aufgaben',
        '3 Punkte-Aufgaben',
        '3-Punkte- Aufgaben',
        '3 Punkte Aufgaben',
    ]
    
    for page_idx, page in enumerate(doc):
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
        for i, text in enumerate(ocr_data['text']):
            if text:
                # Check if this text or nearby texts form the marker
                window = ' '.join([t for t in ocr_data['text'][i:i+5] if t])
                for marker in markers:
                    if marker.replace('-', '').replace(' ', '') in window.replace('-', '').replace(' ', ''):
                        # Found marker! Return page and Y position (in PDF coordinates)
                        # Convert from image coordinates to PDF coordinates
                        y_img = ocr_data['top'][i] + ocr_data['height'][i]
                        y_pdf = y_img / zoom
                        print(f"  Found marker '{marker}' at page {page_idx + 1}, y={y_pdf:.1f}")
                        return page_idx, y_pdf
    
    return None, None


def extract_task_numbers_ocr(page, start_y=0):
    """Extract task numbers from a page using OCR."""
    # Convert page to image
    zoom = 2
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    
    # Convert to PIL Image
    img_data = pix.tobytes('png')
    img = Image.open(io.BytesIO(img_data))
    
    # Run OCR
    ocr_data = pytesseract.image_to_data(img, lang='deu', output_type=pytesseract.Output.DICT)
    
    # Find task numbers (pattern: digit followed by period)
    anchors = []
    pattern = re.compile(r'^(\d+)\.$')
    
    for i, text in enumerate(ocr_data['text']):
        if text:
            match = pattern.match(text.strip())
            if match:
                label = match.group(1)
                
                # Get position (convert from image to PDF coordinates)
                x_img = ocr_data['left'][i]
                y_img = ocr_data['top'][i]
                x_pdf = x_img / zoom
                y_pdf = y_img / zoom
                
                # Skip if before start_y
                if y_pdf < start_y:
                    continue
                
                anchors.append({
                    'label': label,
                    'x_pos': x_pdf,
                    'y_pos': y_pdf,
                })
    
    return anchors


def extract_by_ocr_logic(pdf_path, output_dir):
    """Extract tasks using OCR-based detection with old extraction logic."""
    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Parse filename
    parsed = parse_pdf_filename(pdf_path.name)
    if not parsed:
        print(f"✗ Could not parse filename: {pdf_path.name}")
        return 0
    
    year, class_code = parsed
    print(f"Processing: {year} - {class_code}")
    
    # Open PDF
    doc = fitz.open(pdf_path)
    
    # Find task start marker using OCR
    start_page, start_y = find_task_start_marker_ocr(doc)
    if start_page is None:
        print(f"  ⚠ Could not find task start marker")
        doc.close()
        return 0
    
    # Determine expected task count
    if class_code == '3und4':
        max_tasks = 21
    else:
        max_tasks = 30
    
    expected_labels = [str(i) for i in range(1, max_tasks + 1)]
    
    # Collect all task number anchors from all pages using OCR
    all_anchors = []
    for page_idx in range(start_page, len(doc)):
        page = doc[page_idx]
        page_start_y = start_y if page_idx == start_page else 0
        
        anchors = extract_task_numbers_ocr(page, page_start_y)
        
        # Add page index to anchors
        for anchor in anchors:
            if anchor['label'] in expected_labels:
                anchor['page_idx'] = page_idx
                all_anchors.append(anchor)
    
    if not all_anchors:
        print(f"  ⚠ No task numbers found")
        doc.close()
        return 0
    
    print(f"  Found {len(all_anchors)} potential task labels via OCR")
    
    # Apply old logic: Find most common X position
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
    
    # Main X position
    main_x = max(x_clusters.keys(), key=lambda x: x_clusters[x])
    print(f"  Main X position: {main_x:.1f} ({x_clusters[main_x]} occurrences)")
    
    # Filter anchors near main X
    filtered_anchors = [
        a for a in all_anchors 
        if abs(a['x_pos'] - main_x) <= x_tolerance
    ]
    
    # Sort by document order
    filtered_anchors.sort(key=lambda a: (a['page_idx'], a['y_pos']))
    
    # Remove duplicate labels (keep first occurrence)
    seen_labels = set()
    selected_anchors = []
    for anchor in filtered_anchors:
        if anchor['label'] not in seen_labels:
            selected_anchors.append(anchor)
            seen_labels.add(anchor['label'])
    
    print(f"  Selected {len(selected_anchors)} unique task labels")
    
    # Extract regions as PNGs
    TOP, BOTTOM, SIDE = 6.0, 4.0, 6.0
    count = 0
    
    for idx, anchor in enumerate(selected_anchors):
        page_idx = anchor['page_idx']
        page = doc[page_idx]
        rect = page.rect
        label = anchor['label']
        y0 = anchor['y_pos']
        
        # Determine vertical bounds
        y_top = max(rect.y0, y0 - TOP)
        
        # Find next anchor on same page
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
        
        # Render at 180 DPI
        zoom = 180 / 72
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, clip=clip, alpha=False)
        
        # Save PNG
        item_filename = f"{year}_{class_code}_{label}.png"
        item_path = output_dir / item_filename
        item_path.write_bytes(pix.tobytes('png'))
        
        count += 1
        print(f"  ✓ Created: {item_filename}")
    
    doc.close()
    return count


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python test_ocr.py <pdf_path>")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    output_dir = Path('test_output')
    
    count = extract_by_ocr_logic(pdf_path, output_dir)
    print(f"\n✓ Extracted {count} tasks to {output_dir}/")
