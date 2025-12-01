#!/usr/bin/env python3
"""
Akademische Analyse des Känguru-Datasets für die Studienarbeit.
Erstellt DATASET_ANALYSIS.md mit detaillierten Statistiken und Visualisierungen.
"""

import json
import statistics
import re
import string
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime

# Externe Bibliotheken für linguistische Analyse
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem.snowball import GermanStemmer
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt
    import ssl

    # SSL-Verifizierung für NLTK-Download umgehen (macOS Fix)
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context
    
    # NLTK Ressourcen laden (falls nicht vorhanden)
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        print("⬇️  Lade NLTK Stopwords...")
        nltk.download('stopwords')
    
    HAS_NLP = True
except ImportError:
    print("⚠️  Warnung: NLTK, WordCloud oder Matplotlib nicht gefunden. Linguistische Analyse eingeschränkt.")
    HAS_NLP = False

def load_dataset(json_path: Path) -> list:
    """Lädt das Dataset aus der JSON-Datei."""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def analyze_text_content(text_list: list, output_dir: Path) -> dict:
    """
    Führt eine linguistische Analyse der Aufgabentexte durch.
    Inklusive Stopword-Removal, Stemming und WordCloud-Generierung.
    """
    if not HAS_NLP or not text_list:
        return {'top_words': [], 'wordcloud_path': None}
    
    print("   ...starte linguistische Analyse...")
    
    # 1. Preprocessing Setup
    stop_words = set(stopwords.words('german'))
    # Zusätzliche domänenspezifische Stopwörter
    custom_stops = {'dass', 'wurde', 'wurden', 'gibt', 'beim', 'dabei', 'wer', 'wie', 'was', 'wo', 'wann'}
    stop_words.update(custom_stops)
    
    stemmer = GermanStemmer()
    
    processed_words = []
    
    # 2. Text Processing
    for text in text_list:
        # Lowercase & Punctuation removal
        text = text.lower()
        text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)
        text = re.sub(r'\d+', '', text) # Zahlen entfernen
        
        words = text.split()
        
        for word in words:
            if word not in stop_words and len(word) > 2:
                # Stemming anwenden
                stemmed = stemmer.stem(word)
                processed_words.append(stemmed)
                # Alternativ: Originalwort behalten für bessere Lesbarkeit in WordCloud
                # processed_words.append(word) 
    
    # Wir nutzen hier die Originalwörter (gefiltert) für die WordCloud, 
    # da gestemmte Wörter oft schwer lesbar sind (z.B. "rechn" statt "rechnen")
    # Für die Frequenzanalyse nutzen wir ebenfalls die ungestemmten, aber normalisierten Wörter
    
    display_words = []
    for text in text_list:
        text = text.lower()
        text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)
        text = re.sub(r'\d+', '', text)
        words = text.split()
        display_words.extend([w for w in words if w not in stop_words and len(w) > 2])
    
    word_counts = Counter(display_words)
    top_words = word_counts.most_common(25)
    
    # 3. WordCloud Generierung
    wordcloud = WordCloud(
        width=1600, 
        height=800, 
        background_color='white', 
        colormap='viridis',
        max_words=200
    ).generate_from_frequencies(word_counts)
    
    wc_path = output_dir / "wordcloud.png"
    plt.figure(figsize=(20,10))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(wc_path)
    plt.close()
    
    return {
        'top_words': top_words,
        'wordcloud_path': "wordcloud.png"
    }

def load_solutions(base_dir: Path) -> dict:
    """Lädt die erwarteten Lösungen aus den JSON-Dateien."""
    solutions = {}
    
    # 1998-2011
    path1 = base_dir / "data" / "lösungen_1998_2011.json"
    if path1.exists():
        with open(path1, 'r', encoding='utf-8') as f:
            solutions.update(json.load(f))
            
    # 2012-2025
    path2 = base_dir / "data" / "lösungen_2012_2025.json"
    if path2.exists():
        with open(path2, 'r', encoding='utf-8') as f:
            solutions.update(json.load(f))
            
    return solutions

def calculate_statistics(data: list, base_dir: Path) -> dict:
    """Berechnet umfassende Statistiken für die akademische Analyse."""
    stats = {
        'total_tasks': len(data),
        'years': set(),
        'classes': set(),
        'by_year': defaultdict(int),
        'by_class': defaultdict(int),
        'by_difficulty': defaultdict(int),
        'by_category': defaultdict(int),
        'by_modality': {'text_only': 0, 'visual': 0, 'unknown': 0},
        'by_answer': defaultdict(int),
        'cross_category_class': defaultdict(lambda: defaultdict(int)),
        'cross_modality_category': defaultdict(lambda: defaultdict(int)),
        'cross_modality_difficulty': {'text_only': defaultdict(int), 'visual': defaultdict(int)},
        'cross_modality_answer': {'text_only': defaultdict(int), 'visual': defaultdict(int)},
        'cross_difficulty_category': defaultdict(lambda: defaultdict(int)),
        'text_stats': {'lengths': [], 'options_counts': []},
        'all_questions': [],
        'found_per_year_class': defaultdict(int),
        'completeness': {'missing': [], 'extra': [], 'coverage': 0.0}
    }

    # Load expected tasks from solutions
    expected_tasks = set(load_solutions(base_dir).keys())
    found_tasks = set()

    for entry in data:
        # Basis-Informationen
        year = entry.get('year')
        cls = entry.get('class')
        task_id = entry.get('task_id', '')
        
        # Task Key für Abgleich rekonstruieren
        # Format im Dataset: image_path="dataset_final/1998_3und4_1.png"
        # Format in Lösungen: "1998_3und4_1"
        img_path = entry.get('image_path', '')
        task_key = Path(img_path).stem # "1998_3und4_1"
        found_tasks.add(task_key)
        
        if year:
            stats['years'].add(year)
            stats['by_year'][year] += 1
        
        if cls:
            stats['classes'].add(cls)
            stats['by_class'][cls] += 1
            
        if year and cls:
            stats['found_per_year_class'][(str(year), str(cls))] += 1
        
        # Schwierigkeitsgrad (A/B/C)
        if task_id and len(task_id) > 0:
            diff = task_id[0]
            if diff in ['A', 'B', 'C']:
                stats['by_difficulty'][diff] += 1
                
                # Cross: Difficulty vs Category
                cat = entry.get('math_category', 'unknown')
                if not cat: cat = 'unknown'
                stats['cross_difficulty_category'][diff][cat] += 1
        
        # Antwortverteilung (Bias-Check)
        answer = entry.get('answer')
        if answer and answer in ['A', 'B', 'C', 'D', 'E']:
            stats['by_answer'][answer] += 1
        
        # Kategorie
        cat = entry.get('math_category', 'unknown')
        if not cat: cat = 'unknown'
        stats['by_category'][cat] += 1
        stats['cross_category_class'][cls][cat] += 1
        
        # Modalität
        is_text = entry.get('is_text_only')
        modality_key = None
        
        if is_text is True:
            stats['by_modality']['text_only'] += 1
            stats['cross_modality_category'][cat]['text_only'] += 1
            modality_key = 'text_only'
        elif is_text is False:
            stats['by_modality']['visual'] += 1
            stats['cross_modality_category'][cat]['visual'] += 1
            modality_key = 'visual'
        else:
            stats['by_modality']['unknown'] += 1
            
        if modality_key:
            # Difficulty per Modality
            if task_id and len(task_id) > 0:
                diff = task_id[0]
                if diff in ['A', 'B', 'C']:
                    stats['cross_modality_difficulty'][modality_key][diff] += 1
            
            # Answer per Modality
            if answer and answer in ['A', 'B', 'C', 'D', 'E']:
                stats['cross_modality_answer'][modality_key][answer] += 1
            
        # Text-Statistiken
        extracted = entry.get('extracted_text')
        if extracted and isinstance(extracted, dict):
            q = extracted.get('question', '')
            if q:
                # Wortanzahl
                stats['text_stats']['lengths'].append(len(q.split()))
                stats['all_questions'].append(q)
            
            opts = extracted.get('answer_options', [])
            if opts:
                stats['text_stats']['options_counts'].append(len(opts))

    # Completeness Analysis
    missing = expected_tasks - found_tasks
    extra = found_tasks - expected_tasks
    
    stats['completeness']['missing'] = sorted(list(missing))
    stats['completeness']['extra'] = sorted(list(extra))
    stats['completeness']['expected_count'] = len(expected_tasks)
    stats['completeness']['found_count'] = len(found_tasks)
    
    if len(expected_tasks) > 0:
        stats['completeness']['coverage'] = len(found_tasks & expected_tasks) / len(expected_tasks)

    # Detailed Completeness (Year x Class)
    expected_counts = defaultdict(int)
    for key in expected_tasks:
        # key format: YYYY_Class_ID
        parts = key.split('_')
        if len(parts) >= 2:
            y, c = parts[0], parts[1]
            expected_counts[(y, c)] += 1
            
    detailed_rows = []
    sorted_years = sorted(list(stats['years']), key=lambda x: int(x))
    sorted_classes = ['3und4', '5und6', '7und8', '9und10', '11bis13']
    
    for y in sorted_years:
        for c in sorted_classes:
            exp = expected_counts.get((str(y), c), 0)
            fnd = stats['found_per_year_class'].get((str(y), c), 0)
            if exp > 0:
                detailed_rows.append({
                    'year': y,
                    'class': c,
                    'found': fnd,
                    'expected': exp,
                    'missing': exp - fnd,
                    'rate': (fnd/exp*100)
                })
    stats['detailed_completeness'] = detailed_rows

    # Linguistische Analyse durchführen
    stats['linguistics'] = analyze_text_content(stats['all_questions'], output_dir=base_dir)

    return stats

def generate_markdown_report(stats: dict, output_path: Path):
    """Generiert den Markdown-Bericht auf akademischem Niveau."""
    
    # Hilfsfunktionen für Formatierung
    def pct(value, total):
        return f"{(value / total * 100):.1f}%" if total > 0 else "0.0%"

    total = stats['total_tasks']
    years = sorted(list(stats['years']))
    classes = sorted(list(stats['classes']), key=lambda x: {'3und4': 1, '5und6': 2, '7und8': 3, '9und10': 4, '11bis13': 5}.get(x, 99))
    
    content = []
    content.append("# Analyse des Känguru-Mathematik-Datasets\n")
    content.append(f"*Generiert am {datetime.now().strftime('%d.%m.%Y')}*\n")
    
    # 1. Einleitung
    content.append("## 1. Einleitung und Datensatzbeschreibung\n")
    content.append(f"Der vorliegende Datensatz umfasst insgesamt **{total:,}** Mathematikaufgaben aus dem Känguru-Wettbewerb ")
    content.append(f"der Jahre **{min(years)} bis {max(years)}**. ")
    content.append("Der Datensatz wurde für die Evaluation von Vision-Language-Models (VLMs) kuratiert und enthält ")
    content.append("sowohl die visuellen Repräsentationen (Bilder) als auch semantische Metadaten.\n")
    
    # 2. Temporale und Strukturelle Verteilung
    content.append("## 2. Temporale und Strukturelle Verteilung\n")
    
    # Jahre
    content.append("### 2.1 Zeitliche Abdeckung\n")
    content.append("Die Verteilung der Aufgaben über den Erfassungszeitraum ist wie folgt:\n")
    content.append("| Jahr | Anzahl Aufgaben | Anteil |\n")
    content.append("|------|-----------------|--------|\n")
    for year in years:
        count = stats['by_year'][year]
        content.append(f"| {year} | {count} | {pct(count, total)} |\n")
    
    # Klassenstufen
    content.append("\n### 2.2 Verteilung nach Klassenstufen\n")
    content.append("Der Datensatz deckt alle relevanten Altersgruppen des Wettbewerbs ab:\n")
    content.append("| Klassenstufe | Anzahl Aufgaben | Anteil |\n")
    content.append("|--------------|-----------------|--------|\n")
    class_labels = {'3und4': '3./4. Klasse', '5und6': '5./6. Klasse', '7und8': '7./8. Klasse', '9und10': '9./10. Klasse', '11bis13': '11.-13. Klasse'}
    for cls in classes:
        count = stats['by_class'][cls]
        label = class_labels.get(cls, cls)
        content.append(f"| {label} | {count} | {pct(count, total)} |\n")
        
    # Schwierigkeitsgrad
    content.append("\n### 2.3 Komplexitätsverteilung (Schwierigkeitsgrad)\n")
    content.append("Die Aufgaben sind in drei Schwierigkeitsgrade unterteilt (A: 3 Punkte, B: 4 Punkte, C: 5 Punkte):\n")
    content.append("| Schwierigkeitsgrad | Anzahl Aufgaben | Anteil |\n")
    content.append("|--------------------|-----------------|--------|\n")
    diff_labels = {'A': 'A (Leicht / 3 Pkt)', 'B': 'B (Mittel / 4 Pkt)', 'C': 'C (Schwer / 5 Pkt)'}
    for diff in ['A', 'B', 'C']:
        count = stats['by_difficulty'][diff]
        content.append(f"| {diff_labels[diff]} | {count} | {pct(count, total)} |\n")

    # 3. Inhaltliche Analyse
    content.append("\n## 3. Inhaltliche Analyse (Mathematische Teilgebiete)\n")
    content.append("Die Aufgaben wurden mittels GPT-4o Vision in vier mathematische Hauptkategorien klassifiziert:\n")
    content.append("| Kategorie | Anzahl Aufgaben | Anteil |\n")
    content.append("|-----------|-----------------|--------|\n")
    
    cats = ['Arithmetik', 'Geometrie', 'Algebra', 'Stochastik', 'unknown']
    # Sort cats by count desc
    cats_sorted = sorted(cats, key=lambda x: stats['by_category'][x], reverse=True)
    
    for cat in cats_sorted:
        count = stats['by_category'][cat]
        content.append(f"| {cat} | {count} | {pct(count, total)} |\n")
        
    # Cross-Tab: Category vs Class
    content.append("\n### 3.1 Themenverteilung nach Klassenstufe\n")
    content.append("Die folgende Matrix zeigt die thematischen Schwerpunkte je Altersgruppe:\n")
    content.append("| Klassenstufe | " + " | ".join(cats_sorted[:4]) + " |\n")
    content.append("|--------------|" + "|".join(["---"] * 4) + "|\n")
    
    for cls in classes:
        row = f"| {class_labels.get(cls, cls)} |"
        cls_total = stats['by_class'][cls]
        for cat in cats_sorted[:4]:
            count = stats['cross_category_class'][cls][cat]
            # row += f" {count} ({pct(count, cls_total)}) |" # Too wide maybe?
            row += f" {count} |"
        content.append(row + "\n")

    # 4. Modalitätsanalyse
    content.append("\n## 4. Modalitätsanalyse (Text vs. Visuell)\n")
    content.append("Ein zentraler Aspekt für die VLM-Evaluation ist die Unterscheidung zwischen Aufgaben, ")
    content.append("die rein textbasiert lösbar sind, und solchen, die visuelle Elemente erfordern.\n")
    
    t_only = stats['by_modality']['text_only']
    vis = stats['by_modality']['visual']
    unk = stats['by_modality']['unknown']
    analyzed_total = t_only + vis
    
    content.append(f"- **Text-Only Aufgaben**: {t_only} ({pct(t_only, analyzed_total)} der analysierten Aufgaben)\n")
    content.append(f"- **Visuell-Notwendige Aufgaben**: {vis} ({pct(vis, analyzed_total)} der analysierten Aufgaben)\n")
    if unk > 0:
        content.append(f"- *Nicht analysiert*: {unk}\n")
        
    # Cross-Tab: Modality vs Category
    content.append("\n### 4.1 Visuelle Abhängigkeit nach Kategorie\n")
    content.append("| Kategorie | Text-Only | Visuell | Visueller Anteil |\n")
    content.append("|-----------|-----------|---------|------------------|\n")
    
    for cat in cats_sorted[:4]:
        t = stats['cross_modality_category'][cat]['text_only']
        v = stats['cross_modality_category'][cat]['visual']
        cat_total = t + v
        vis_rate = (v / cat_total * 100) if cat_total > 0 else 0
        content.append(f"| {cat} | {t} | {v} | **{vis_rate:.1f}%** |\n")
        
    # 4.2 Visuelle Aufgaben: Schwierigkeit und Antwortverteilung
    content.append("\n### 4.2 Visuelle Aufgaben: Schwierigkeit und Antwortverteilung\n")
    content.append("Um sicherzustellen, dass visuelle Aufgaben nicht systematisch schwerer oder leichter sind, ")
    content.append("wird hier die Verteilung für die Teilmenge der visuell-notwendigen Aufgaben betrachtet.\n")
    
    # Difficulty Table
    content.append("#### Schwierigkeitsgrad (Visuell)\n")
    content.append("| Schwierigkeitsgrad | Anzahl (Visuell) | Anteil |\n")
    content.append("|--------------------|------------------|--------|\n")
    
    vis_diff_stats = stats['cross_modality_difficulty']['visual']
    total_vis_diff = sum(vis_diff_stats.values())
    
    for diff in ['A', 'B', 'C']:
        count = vis_diff_stats[diff]
        content.append(f"| {diff_labels[diff]} | {count} | {pct(count, total_vis_diff)} |\n")
        
    # Answer Table
    content.append("\n#### Antwortverteilung (Visuell)\n")
    content.append("| Antwort | Anzahl (Visuell) | Anteil |\n")
    content.append("|---------|------------------|--------|\n")
    
    vis_ans_stats = stats['cross_modality_answer']['visual']
    total_vis_ans = sum(vis_ans_stats.values())
    
    for ans in ['A', 'B', 'C', 'D', 'E']:
        count = vis_ans_stats[ans]
        content.append(f"| {ans} | {count} | {pct(count, total_vis_ans)} |\n")

    # Category Table (Visual)
    content.append("\n#### Mathematische Kategorien (Visuell)\n")
    content.append("| Kategorie | Anzahl (Visuell) | Anteil |\n")
    content.append("|-----------|------------------|--------|\n")
    
    # Calculate total visual tasks for categories
    total_vis_cat = 0
    vis_cat_counts = {}
    for cat in cats_sorted[:4]:
        c = stats['cross_modality_category'][cat]['visual']
        vis_cat_counts[cat] = c
        total_vis_cat += c
        
    for cat in cats_sorted[:4]:
        count = vis_cat_counts[cat]
        content.append(f"| {cat} | {count} | {pct(count, total_vis_cat)} |\n")

    # 5. Textuelle Eigenschaften
    content.append("\n## 5. Textuelle Eigenschaften\n")
    lengths = stats['text_stats']['lengths']
    if lengths:
        avg_len = statistics.mean(lengths)
        median_len = statistics.median(lengths)
        content.append(f"- **Durchschnittliche Wortanzahl pro Frage**: {avg_len:.1f} Wörter\n")
        content.append(f"- **Median der Wortanzahl**: {median_len:.1f} Wörter\n")
        content.append(f"- **Min/Max**: {min(lengths)} / {max(lengths)} Wörter\n")
    
    # 6. Antwortverteilung (Bias-Check)
    content.append("\n## 6. Antwortverteilung (Bias-Check)\n")
    content.append("Um sicherzustellen, dass das Modell nicht durch Raten einer häufigsten Antwort (z.B. immer 'C') ")
    content.append("eine künstlich hohe Performance erzielt, wurde die Verteilung der korrekten Antwortbuchstaben analysiert.\n")
    
    content.append("| Antwort | Anzahl | Anteil |\n")
    content.append("|---------|--------|--------|\n")
    
    total_answers = sum(stats['by_answer'].values())
    for ans in ['A', 'B', 'C', 'D', 'E']:
        count = stats['by_answer'][ans]
        content.append(f"| {ans} | {count} | {pct(count, total_answers)} |\n")
        
    # 7. Linguistische Analyse
    content.append("\n## 7. Linguistische Analyse\n")
    content.append("### 7.1 Methodik\n")
    content.append("Für die Analyse der häufigsten Begriffe wurden folgende Vorverarbeitungsschritte durchgeführt:\n")
    content.append("1. **Normalisierung**: Umwandlung in Kleinschreibung.\n")
    content.append("2. **Bereinigung**: Entfernung von Satzzeichen und Zahlen.\n")
    content.append("3. **Stopword-Removal**: Entfernung häufiger deutscher Füllwörter (basierend auf NLTK 'german' corpus) sowie domänenspezifischer Begriffe (z.B. 'dass', 'wurde').\n")
    content.append("4. **Filterung**: Ausschluss von Wörtern mit weniger als 3 Buchstaben.\n")
    
    content.append("\n### 7.2 Häufigste Begriffe (Top 25)\n")
    content.append("| Rang | Begriff | Häufigkeit |\n")
    content.append("|------|---------|------------|\n")
    
    top_words = stats.get('linguistics', {}).get('top_words', [])
    for i, (word, count) in enumerate(top_words, 1):
        content.append(f"| {i} | {word} | {count} |\n")
        
    wc_path = stats.get('linguistics', {}).get('wordcloud_path')
    if wc_path:
        content.append("\n### 7.3 WordCloud\n")
        content.append(f"![WordCloud der häufigsten Begriffe]({wc_path})\n")
    
    # 8. Vollständigkeitsanalyse
    content.append("\n## 8. Vollständigkeitsanalyse\n")
    comp = stats.get('completeness', {})
    expected = comp.get('expected_count', 0)
    found = comp.get('found_count', 0)
    coverage = comp.get('coverage', 0)
    missing = comp.get('missing', [])
    extra = comp.get('extra', [])
    
    content.append(f"Der Datensatz wurde gegen die offiziellen Lösungsschlüssel abgeglichen, um die Vollständigkeit zu verifizieren.\n")
    content.append(f"- **Erwartete Aufgaben**: {expected}\n")
    content.append(f"- **Gefundene Aufgaben**: {found}\n")
    content.append(f"- **Abdeckung (Coverage)**: {coverage*100:.1f}%\n")
    
    if missing:
        content.append(f"\n### 8.1 Fehlende Aufgaben ({len(missing)})\n")
        content.append("Folgende Aufgaben sind im Lösungsschlüssel verzeichnet, konnten aber nicht extrahiert werden (z.B. aufgrund von PDF-Fehlern oder fehlenden Seiten):\n")
        # Zeige nur die ersten 10 und die letzten 10 wenn es viele sind
        if len(missing) > 20:
            content.append(", ".join(missing[:10]) + " ... " + ", ".join(missing[-10:]) + "\n")
        else:
            content.append(", ".join(missing) + "\n")
            
    # 8.2 Detaillierte Abdeckung nach Jahr und Klasse
    content.append("\n### 8.2 Detaillierte Abdeckung nach Jahr und Klasse\n")
    content.append("Die folgende Tabelle zeigt die Anzahl der extrahierten Aufgaben im Vergleich zu den erwarteten Aufgaben für jede Jahrgangsstufe und jedes Jahr.\n")
    content.append("| Jahr | Klasse | Gefunden | Erwartet | Fehlend | Quote |\n")
    content.append("| :--- | :--- | :--- | :--- | :--- | :--- |\n")
    
    for row in stats.get('detailed_completeness', []):
        y = row['year']
        c = row['class']
        f = row['found']
        e = row['expected']
        m = row['missing']
        r = row['rate']
        
        # Highlight rows with missing items or 0% coverage
        y_str = f"**{y}**" if m > 0 else str(y)
        c_str = f"**{c}**" if r < 10.0 else str(c)
        m_str = f"**{m}**" if m > 0 else str(m)
        r_str = f"**{r:.1f}%**" if r < 10.0 else f"{r:.1f}%"
        
        # Only bold the whole row if it's really bad (0%)
        if r == 0:
             content.append(f"| {y_str} | **{c}** | {f} | **{e}** | **{m}** | **{r:.1f}%** |\n")
        else:
             content.append(f"| {y_str} | {c} | {f} | {e} | {m_str} | {r_str} |\n")

    if extra:
        content.append(f"\n### 8.3 Zusätzliche Aufgaben ({len(extra)})\n")
        content.append("Folgende Aufgaben wurden extrahiert, sind aber nicht im Lösungsschlüssel enthalten (z.B. Aufgaben aus Jahren ohne Lösungsdaten):\n")
        if len(extra) > 20:
            content.append(", ".join(extra[:10]) + " ... " + ", ".join(extra[-10:]) + "\n")
        else:
            content.append(", ".join(extra) + "\n")

    content.append("\n---\n")
    content.append("Diese Analyse dient als Grundlage für die differenzierte Auswertung der Modell-Leistungen in der Studienarbeit.")
    
    output_path.write_text(''.join(content), encoding='utf-8')

def main():
    base_dir = Path(__file__).parent.parent
    json_path = base_dir / "dataset_final.json"
    output_path = base_dir / "DATASET_ANALYSIS.md"
    
    print(f"📂 Lade Dataset von {json_path}...")
    if not json_path.exists():
        print("❌ Datei nicht gefunden!")
        return
        
    data = load_dataset(json_path)
    print(f"✅ {len(data)} Einträge geladen.")
    
    print("📊 Berechne Statistiken...")
    stats = calculate_statistics(data, base_dir)
    
    print("📝 Generiere Bericht...")
    generate_markdown_report(stats, output_path)
    
    print(f"✅ Analyse abgeschlossen! Bericht gespeichert unter:\n   {output_path}")

if __name__ == "__main__":
    main()
