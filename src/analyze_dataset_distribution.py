#!/usr/bin/env python3
"""
Analyzes the distribution of tasks in dataset_final and dataset_final_not_used folders.
Shows distribution by year, class level, and task difficulty (A/B/C categories).
"""

from pathlib import Path
from collections import defaultdict
import re

def parse_filename(filename: str) -> dict | None:
    """
    Parse filename to extract year, class, and task info.
    Format: YYYY_class_taskID.png
    Examples: 2024_11bis13_A1.png, 2012_7und8_B5.png
    """
    pattern = r'(\d{4})_([^_]+)_([ABC]\d+)\.png'
    match = re.match(pattern, filename)
    
    if match:
        year, class_level, task_id = match.groups()
        difficulty = task_id[0]  # A, B, or C
        task_number = int(task_id[1:])  # numeric part
        
        return {
            'year': int(year),
            'class': class_level,
            'task_id': task_id,
            'difficulty': difficulty,
            'task_number': task_number
        }
    return None

def analyze_directory(directory_path: Path) -> dict:
    """Analyze all PNG files in a directory."""
    stats = {
        'total_files': 0,
        'by_year': defaultdict(int),
        'by_class': defaultdict(int),
        'by_difficulty': defaultdict(int),
        'by_year_class': defaultdict(lambda: defaultdict(int)),
        'by_year_difficulty': defaultdict(lambda: defaultdict(int)),
        'years': set(),
        'classes': set(),
    }
    
    if not directory_path.exists():
        return stats
    
    for file_path in directory_path.glob('*.png'):
        parsed = parse_filename(file_path.name)
        
        if parsed:
            stats['total_files'] += 1
            stats['by_year'][parsed['year']] += 1
            stats['by_class'][parsed['class']] += 1
            stats['by_difficulty'][parsed['difficulty']] += 1
            stats['by_year_class'][parsed['year']][parsed['class']] += 1
            stats['by_year_difficulty'][parsed['year']][parsed['difficulty']] += 1
            stats['years'].add(parsed['year'])
            stats['classes'].add(parsed['class'])
    
    return stats

def print_distribution(stats: dict, title: str):
    """Print distribution statistics in a formatted way."""
    print(f"\n{'='*80}")
    print(f"{title:^80}")
    print(f"{'='*80}\n")
    
    print(f"📊 Gesamt: {stats['total_files']} Aufgaben\n")
    
    # Distribution by year
    print("📅 Verteilung nach Jahren:")
    print("-" * 60)
    if stats['by_year']:
        years_sorted = sorted(stats['by_year'].keys())
        for year in years_sorted:
            count = stats['by_year'][year]
            percentage = (count / stats['total_files'] * 100) if stats['total_files'] > 0 else 0
            bar = '█' * int(percentage / 2)
            print(f"  {year}: {count:4d} ({percentage:5.1f}%) {bar}")
        
        # Year statistics
        counts = list(stats['by_year'].values())
        avg = sum(counts) / len(counts) if counts else 0
        print(f"\n  📈 Durchschnitt pro Jahr: {avg:.1f}")
        print(f"  📉 Min: {min(counts)} | Max: {max(counts)} | Spanne: {max(counts) - min(counts)}")
    else:
        print("  Keine Daten")
    
    # Distribution by class level
    print(f"\n🎓 Verteilung nach Klassenstufen:")
    print("-" * 60)
    if stats['by_class']:
        class_order = ['3und4', '5und6', '7und8', '9und10', '11bis13']
        for class_level in class_order:
            if class_level in stats['by_class']:
                count = stats['by_class'][class_level]
                percentage = (count / stats['total_files'] * 100) if stats['total_files'] > 0 else 0
                bar = '█' * int(percentage / 2)
                print(f"  {class_level:10s}: {count:4d} ({percentage:5.1f}%) {bar}")
    else:
        print("  Keine Daten")
    
    # Distribution by difficulty
    print(f"\n⭐ Verteilung nach Schwierigkeitsgrad:")
    print("-" * 60)
    if stats['by_difficulty']:
        difficulty_labels = {'A': 'A (Leicht)', 'B': 'B (Mittel)', 'C': 'C (Schwer)'}
        for diff in ['A', 'B', 'C']:
            if diff in stats['by_difficulty']:
                count = stats['by_difficulty'][diff]
                percentage = (count / stats['total_files'] * 100) if stats['total_files'] > 0 else 0
                bar = '█' * int(percentage / 2)
                print(f"  {difficulty_labels[diff]:12s}: {count:4d} ({percentage:5.1f}%) {bar}")
    else:
        print("  Keine Daten")
    
    # Year x Class matrix
    if stats['by_year_class']:
        print(f"\n📋 Detaillierte Verteilung (Jahr × Klassenstufe):")
        print("-" * 80)
        
        class_order = ['3und4', '5und6', '7und8', '9und10', '11bis13']
        years_sorted = sorted(stats['by_year_class'].keys())
        
        # Header
        print(f"  {'Jahr':<6}", end='')
        for class_level in class_order:
            print(f"{class_level:>10}", end='')
        print(f"  {'Gesamt':>10}")
        print("  " + "-" * 76)
        
        # Rows
        for year in years_sorted:
            print(f"  {year:<6}", end='')
            year_total = 0
            for class_level in class_order:
                count = stats['by_year_class'][year].get(class_level, 0)
                year_total += count
                if count > 0:
                    print(f"{count:>10}", end='')
                else:
                    print(f"{'—':>10}", end='')
            print(f"  {year_total:>10}")
        
        # Footer totals
        print("  " + "-" * 76)
        print(f"  {'Total':<6}", end='')
        for class_level in class_order:
            total = sum(stats['by_year_class'][year].get(class_level, 0) for year in years_sorted)
            print(f"{total:>10}", end='')
        print(f"  {stats['total_files']:>10}")

def compare_datasets(stats1: dict, stats2: dict):
    """Compare two datasets and show differences."""
    print(f"\n{'='*80}")
    print(f"{'VERGLEICH: dataset_final vs. dataset_final_not_used':^80}")
    print(f"{'='*80}\n")
    
    total1 = stats1['total_files']
    total2 = stats2['total_files']
    
    print(f"📊 Gesamtübersicht:")
    print(f"  • dataset_final:          {total1:4d} Aufgaben (für Evaluation)")
    print(f"  • dataset_final_not_used: {total2:4d} Aufgaben (nicht tauglich)")
    print(f"  • Gesamt verfügbar:       {total1 + total2:4d} Aufgaben")
    print(f"  • Nutzungsrate:           {total1 / (total1 + total2) * 100:.1f}%")
    
    # Compare years
    all_years = sorted(set(stats1['years']) | set(stats2['years']))
    if all_years:
        print(f"\n📅 Jahresvergleich:")
        print(f"  {'Jahr':<6} {'Genutzt':>10} {'Nicht genutzt':>15} {'Total':>10} {'Rate':>8}")
        print("  " + "-" * 56)
        
        for year in all_years:
            used = stats1['by_year'].get(year, 0)
            unused = stats2['by_year'].get(year, 0)
            total = used + unused
            rate = (used / total * 100) if total > 0 else 0
            print(f"  {year:<6} {used:>10} {unused:>15} {total:>10} {rate:>7.1f}%")
    
    # Compare classes
    all_classes = ['3und4', '5und6', '7und8', '9und10', '11bis13']
    print(f"\n🎓 Klassenstufenvergleich:")
    print(f"  {'Klasse':<12} {'Genutzt':>10} {'Nicht genutzt':>15} {'Total':>10} {'Rate':>8}")
    print("  " + "-" * 62)
    
    for class_level in all_classes:
        used = stats1['by_class'].get(class_level, 0)
        unused = stats2['by_class'].get(class_level, 0)
        total = used + unused
        rate = (used / total * 100) if total > 0 else 0
        if total > 0:
            print(f"  {class_level:<12} {used:>10} {unused:>15} {total:>10} {rate:>7.1f}%")

def main():
    project_root = Path(__file__).parent.parent
    
    dataset_final = project_root / 'data' / 'dataset_final'
    dataset_not_used = project_root / 'data' / 'dataset_final_not_used'
    
    print("\n🔍 Analysiere Datensätze...\n")
    
    # Analyze both directories
    stats_final = analyze_directory(dataset_final)
    stats_not_used = analyze_directory(dataset_not_used)
    
    # Print individual statistics
    print_distribution(stats_final, "DATASET_FINAL (Für VLM Evaluation)")
    print_distribution(stats_not_used, "DATASET_FINAL_NOT_USED (Nicht tauglich)")
    
    # Compare datasets
    compare_datasets(stats_final, stats_not_used)
    
    # Insights
    print(f"\n{'='*80}")
    print(f"{'💡 ERKENNTNISSE':^80}")
    print(f"{'='*80}\n")
    
    if stats_final['total_files'] > 0:
        # Year with most/least tasks
        years = stats_final['by_year']
        if years:
            max_year = max(years.items(), key=lambda x: x[1])
            min_year = min(years.items(), key=lambda x: x[1])
            print(f"📈 Jahr mit den meisten Aufgaben: {max_year[0]} ({max_year[1]} Aufgaben)")
            print(f"📉 Jahr mit den wenigsten Aufgaben: {min_year[0]} ({min_year[1]} Aufgaben)")
        
        # Class balance
        classes = stats_final['by_class']
        if classes:
            max_class = max(classes.items(), key=lambda x: x[1])
            min_class = min(classes.items(), key=lambda x: x[1])
            balance_ratio = min_class[1] / max_class[1] if max_class[1] > 0 else 0
            print(f"\n🎓 Klassenstufe mit den meisten Aufgaben: {max_class[0]} ({max_class[1]} Aufgaben)")
            print(f"🎓 Klassenstufe mit den wenigsten Aufgaben: {min_class[0]} ({min_class[1]} Aufgaben)")
            print(f"⚖️  Balance-Verhältnis: {balance_ratio:.2f} (1.0 = perfekt ausgewogen)")
        
        # Difficulty balance
        difficulties = stats_final['by_difficulty']
        if difficulties:
            print(f"\n⭐ Schwierigkeitsverteilung:")
            total = sum(difficulties.values())
            for diff in ['A', 'B', 'C']:
                if diff in difficulties:
                    count = difficulties[diff]
                    pct = count / total * 100
                    print(f"   {diff}: {count} ({pct:.1f}%)")
    
    print(f"\n{'='*80}\n")

if __name__ == '__main__':
    main()
