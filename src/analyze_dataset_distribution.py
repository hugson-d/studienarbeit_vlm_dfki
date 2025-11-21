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
    Examples: 2024_11bis13_A1.png, 2012_7und8_B5.png, 2010_3und4_15.png
    """
    # Try format with A/B/C difficulty (2012-2025)
    pattern_abc = r'(\d{4})_([^_]+)_([ABC]\d+)\.png'
    match = re.match(pattern_abc, filename)
    
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
    
    # Try format with numeric ID only (1998-2011)
    pattern_numeric = r'(\d{4})_([^_]+)_(\d+)\.png'
    match = re.match(pattern_numeric, filename)
    
    if match:
        year, class_level, task_id = match.groups()
        task_num = int(task_id)
        
        # Map numeric IDs to difficulty levels for 1998-2011
        # Klassenstufen 3und4 und 5und6: 1-8=A, 9-16=B, 17-24=C
        # Klassenstufen 7und8, 9und10, 11bis13: 1-10=A, 11-20=B, 21-30=C
        if class_level in ['3und4', '5und6']:
            if task_num <= 8:
                difficulty = 'A'
            elif task_num <= 16:
                difficulty = 'B'
            else:  # 17-24
                difficulty = 'C'
        else:  # 7und8, 9und10, 11bis13
            if task_num <= 10:
                difficulty = 'A'
            elif task_num <= 20:
                difficulty = 'B'
            else:  # 21-30
                difficulty = 'C'
        
        return {
            'year': int(year),
            'class': class_level,
            'task_id': task_id,
            'difficulty': difficulty,
            'task_number': task_num
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

def generate_readme(stats_final: dict, stats_not_used: dict, readme_path: Path):
    """Generate DATASET_STATS.md with all distribution information."""
    from datetime import datetime
    
    content = []
    content.append("# Känguru-Wettbewerb Dataset Statistiken\n")
    content.append(f"*Automatisch generiert am {datetime.now().strftime('%d.%m.%Y um %H:%M Uhr')}*\n")
    content.append("---\n")
    
    # Overview
    total_final = stats_final['total_files']
    total_not_used = stats_not_used['total_files']
    total_all = total_final + total_not_used
    usage_rate = (total_final / total_all * 100) if total_all > 0 else 0
    
    content.append("## 📊 Übersicht\n")
    content.append(f"- **Gesamt verfügbare Aufgaben**: {total_all:,}\n")
    content.append(f"- **Verwendbare Aufgaben** (dataset_final): {total_final:,}\n")
    content.append(f"- **Nicht verwendbare Aufgaben** (dataset_final_not_used): {total_not_used:,}\n")
    content.append(f"- **Nutzungsrate**: {usage_rate:.1f}%\n")
    content.append(f"- **Zeitraum**: {min(stats_final['years'])}–{max(stats_final['years'])}\n")
    content.append(f"- **Klassenstufen**: 3-4, 5-6, 7-8, 9-10, 11-13\n\n")
    
    # Year distribution
    content.append("## 📅 Verteilung nach Jahren\n")
    content.append("| Jahr | Dataset Final | Not Used | Gesamt | Nutzungsrate |\n")
    content.append("|------|--------------|----------|--------|-------------|\n")
    
    all_years = sorted(set(stats_final['years']) | set(stats_not_used['years']))
    for year in all_years:
        used = stats_final['by_year'].get(year, 0)
        unused = stats_not_used['by_year'].get(year, 0)
        total = used + unused
        rate = (used / total * 100) if total > 0 else 0
        content.append(f"| {year} | {used} | {unused} | {total} | {rate:.1f}% |\n")
    
    # Class distribution
    content.append("\n## 🎓 Verteilung nach Klassenstufen\n")
    content.append("| Klassenstufe | Dataset Final | Not Used | Gesamt | Nutzungsrate |\n")
    content.append("|--------------|--------------|----------|--------|-------------|\n")
    
    class_order = ['3und4', '5und6', '7und8', '9und10', '11bis13']
    class_labels = {'3und4': '3-4', '5und6': '5-6', '7und8': '7-8', '9und10': '9-10', '11bis13': '11-13'}
    
    for class_level in class_order:
        used = stats_final['by_class'].get(class_level, 0)
        unused = stats_not_used['by_class'].get(class_level, 0)
        total = used + unused
        rate = (used / total * 100) if total > 0 else 0
        label = class_labels[class_level]
        content.append(f"| Klasse {label} | {used} | {unused} | {total} | {rate:.1f}% |\n")
    
    # Difficulty distribution
    content.append("\n## ⭐ Verteilung nach Schwierigkeitsgrad\n")
    content.append("| Schwierigkeit | Dataset Final | Not Used | Gesamt | Anteil (Final) |\n")
    content.append("|---------------|--------------|----------|--------|----------------|\n")
    
    difficulty_labels = {'A': 'A (Leicht)', 'B': 'B (Mittel)', 'C': 'C (Schwer)'}
    for diff in ['A', 'B', 'C']:
        used = stats_final['by_difficulty'].get(diff, 0)
        unused = stats_not_used['by_difficulty'].get(diff, 0)
        total = used + unused
        pct = (used / total_final * 100) if total_final > 0 else 0
        content.append(f"| {difficulty_labels[diff]} | {used} | {unused} | {total} | {pct:.1f}% |\n")
    
    # Detailed matrix
    content.append("\n## 📋 Detaillierte Verteilung (Jahr × Klassenstufe)\n")
    content.append("### Dataset Final\n")
    content.append("| Jahr | Klasse 3-4 | Klasse 5-6 | Klasse 7-8 | Klasse 9-10 | Klasse 11-13 | Gesamt |\n")
    content.append("|------|-----------|-----------|-----------|------------|-------------|--------|\n")
    
    years_sorted = sorted(stats_final['by_year_class'].keys())
    for year in years_sorted:
        row = f"| {year} |"
        year_total = 0
        for class_level in class_order:
            count = stats_final['by_year_class'][year].get(class_level, 0)
            year_total += count
            row += f" {count} |"
        row += f" {year_total} |"
        content.append(row + "\n")
    
    # Totals
    content.append("| **Total** |")
    for class_level in class_order:
        total = sum(stats_final['by_year_class'][year].get(class_level, 0) for year in years_sorted)
        content.append(f" **{total}** |")
    content.append(f" **{total_final}** |\n")
    
    # Insights
    content.append("\n## 💡 Erkenntnisse\n")
    
    if stats_final['total_files'] > 0:
        years = stats_final['by_year']
        if years:
            max_year = max(years.items(), key=lambda x: x[1])
            min_year = min(years.items(), key=lambda x: x[1])
            content.append(f"- **Jahr mit den meisten Aufgaben**: {max_year[0]} ({max_year[1]} Aufgaben)\n")
            content.append(f"- **Jahr mit den wenigsten Aufgaben**: {min_year[0]} ({min_year[1]} Aufgaben)\n")
        
        classes = stats_final['by_class']
        if classes:
            max_class = max(classes.items(), key=lambda x: x[1])
            min_class = min(classes.items(), key=lambda x: x[1])
            balance_ratio = min_class[1] / max_class[1] if max_class[1] > 0 else 0
            content.append(f"- **Klassenstufe mit den meisten Aufgaben**: {class_labels[max_class[0]]} ({max_class[1]} Aufgaben)\n")
            content.append(f"- **Klassenstufe mit den wenigsten Aufgaben**: {class_labels[min_class[0]]} ({min_class[1]} Aufgaben)\n")
            content.append(f"- **Balance-Verhältnis**: {balance_ratio:.2f} (1.0 = perfekt ausgewogen)\n")
        
        difficulties = stats_final['by_difficulty']
        if difficulties:
            content.append(f"\n### Schwierigkeitsverteilung (Dataset Final)\n")
            total = sum(difficulties.values())
            for diff in ['A', 'B', 'C']:
                if diff in difficulties:
                    count = difficulties[diff]
                    pct = count / total * 100
                    content.append(f"- **{difficulty_labels[diff]}**: {count} ({pct:.1f}%)\n")
    
    # Mapping information
    content.append("\n## 🔄 Schwierigkeitsgrad-Mapping (1998-2011)\n")
    content.append("Für Aufgaben von 1998-2011 (numerische Task-IDs) gilt:\n\n")
    content.append("**Klassenstufen 3-4 und 5-6:**\n")
    content.append("- A (Leicht): Aufgaben 1-8\n")
    content.append("- B (Mittel): Aufgaben 9-16\n")
    content.append("- C (Schwer): Aufgaben 17-24\n\n")
    content.append("**Klassenstufen 7-8, 9-10 und 11-13:**\n")
    content.append("- A (Leicht): Aufgaben 1-10\n")
    content.append("- B (Mittel): Aufgaben 11-20\n")
    content.append("- C (Schwer): Aufgaben 21-30\n\n")
    content.append("**Ab 2012:** Task-IDs enthalten bereits den Schwierigkeitsgrad (z.B. A1, B15, C30)\n")
    
    # Write to file
    readme_path.write_text(''.join(content), encoding='utf-8')

def main():
    project_root = Path(__file__).parent.parent
    
    dataset_final = project_root / 'data' / 'dataset_final'
    dataset_not_used = project_root / 'data' / 'dataset_final_not_used'
    readme_path = project_root / 'DATASET_STATS.md'
    
    print("\n🔍 Analysiere Datensätze...\n")
    
    # Analyze both directories
    stats_final = analyze_directory(dataset_final)
    stats_not_used = analyze_directory(dataset_not_used)
    
    # Print individual statistics
    print_distribution(stats_final, "DATASET_FINAL (Für VLM Evaluation)")
    print_distribution(stats_not_used, "DATASET_FINAL_NOT_USED (Nicht tauglich)")
    
    # Compare datasets
    compare_datasets(stats_final, stats_not_used)
    
    # Generate README
    print("\n📝 Generiere DATASET_STATS.md...\n")
    generate_readme(stats_final, stats_not_used, readme_path)
    print(f"✅ DATASET_STATS.md erfolgreich aktualisiert!\n")
    
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
