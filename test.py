"""
Test-Skript zum Extrahieren von Aufgaben aus einer einzelnen PDF.

Usage:
    python test.py <path_to_pdf>
    
Example:
    python test.py data/Klasse3-4/kaenguru2007_34.pdf
"""

import sys
import re
from pathlib import Path
from typing import Optional

import fitz  # type: ignore[import]


def parse_pdf_filename(filename: str) -> Optional[tuple]:
    """Extract year and class from filename like 'kaenguru2007_78.pdf'."""
    match = re.match(r'kaenguru(\d{4})_(\d+)\.pdf$', filename)
    if not match:
        return None
    
    year = int(match.group(1))
    class_code = match.group(2)
    
    # Map class code to standard format
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


def extract_items_from_pdf(pdf_path, year, class_code, output_dir):
    """Extract individual task items from old-format PDF (1998-2011)."""
    doc = fitz.open(pdf_path)
    
    # Find where tasks start (after marker)
    start_page, start_y = find_task_start_marker(doc)
    if start_page is None:
        print(f"  ⚠ Could not find task start marker")
        doc.close()
        return 0
    
    print(f"  Found task start at page {start_page + 1}, y={start_y:.1f}")
    
    # Expected task numbers for 1998-2011 format
    # Klasse 3-4: 21 tasks (for most years)
    # Klasse 5-6, 7-8, 9-10, 11-13: 30 tasks each
    if class_code in ['3und4']:
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
        return 0
    
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
        item_filename = f"{year}_{class_code}_{label}.png"
        item_path = output_dir / item_filename
        item_path.parent.mkdir(parents=True, exist_ok=True)
        item_path.write_bytes(pix.tobytes('png'))
        
        print(f"  ✓ Created: {item_filename}")
    
    doc.close()
    return len(selected_anchors)


def main():
    """Main function."""
    if len(sys.argv) != 2:
        print("Usage: python test.py <path_to_pdf>")
        print("Example: python test.py data/Klasse3-4/kaenguru2007_34.pdf")
        sys.exit(1)
    
    pdf_path = Path(sys.argv[1])
    
    if not pdf_path.exists():
        print(f"❌ Error: PDF file not found: {pdf_path}")
        sys.exit(1)
    
    if not pdf_path.suffix.lower() == '.pdf':
        print(f"❌ Error: File is not a PDF: {pdf_path}")
        sys.exit(1)
    
    # Parse filename
    result = parse_pdf_filename(pdf_path.name)
    if not result:
        print(f"❌ Error: Invalid filename format. Expected format: kaenguruYYYY_CC.pdf")
        print(f"   Example: kaenguru2007_34.pdf, kaenguru2010_1113.pdf")
        sys.exit(1)
    
    year, class_code = result
    
    # Create output directory
    output_dir = Path('test_output')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📄 Processing: {pdf_path.name}")
    print(f"   Year: {year}")
    print(f"   Class: {class_code}")
    print(f"   Output: {output_dir}/\n")
    
    # Extract tasks
    count = extract_items_from_pdf(pdf_path, year, class_code, output_dir)
    
    print(f"\n✅ Extraction complete!")
    print(f"   Created {count} PNG files in {output_dir}/")


if __name__ == '__main__':
    main()
