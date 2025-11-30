#!/usr/bin/env python3
"""
Clustering-Analyse der Aufgabentexte.
Untersucht, ob die Texte (TF-IDF) Cluster bilden, die mit den mathematischen Kategorien korrelieren.
"""

import json
import re
import string
import ssl
from pathlib import Path
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt

# Versuch, notwendige Bibliotheken zu importieren
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem.snowball import GermanStemmer
    
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    from sklearn.metrics import homogeneity_score, completeness_score, v_measure_score
    
    import seaborn as sns
except ImportError as e:
    print(f"❌ Fehlende Bibliothek: {e}")
    print("Bitte installiere die fehlenden Pakete: pip install scikit-learn seaborn nltk")
    exit(1)

# SSL Fix für NLTK auf macOS
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# NLTK Setup
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    print("⬇️  Lade NLTK Stopwords...")
    nltk.download('stopwords')

def load_data(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def preprocess_text(text, stop_words, stemmer):
    if not text:
        return ""
    
    # Lowercase
    text = text.lower()
    
    # Remove punctuation and numbers
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)
    text = re.sub(r'\d+', '', text)
    
    words = text.split()
    
    # Stopword removal and stemming
    processed = []
    for w in words:
        if w not in stop_words and len(w) > 2:
            processed.append(stemmer.stem(w))
            
    return " ".join(processed)

def main():
    base_dir = Path(__file__).parent.parent
    json_path = base_dir / "dataset_final.json"
    output_plot = base_dir / "clustering_analysis.png"
    
    print(f"📂 Lade Daten von {json_path}...")
    data = load_data(json_path)
    
    # Daten vorbereiten
    texts = []
    categories = []
    ids = []
    
    stop_words = set(stopwords.words('german'))
    custom_stops = {'dass', 'wurde', 'wurden', 'gibt', 'beim', 'dabei', 'wer', 'wie', 'was', 'wo', 'wann', 'welche', 'welches', 'welchen'}
    stop_words.update(custom_stops)
    stemmer = GermanStemmer()
    
    print("🔄 Verarbeite Texte (Preprocessing)...")
    for entry in data:
        # Nur Einträge mit Kategorie und Text nutzen
        cat = entry.get('math_category')
        extracted = entry.get('extracted_text', {})
        question = extracted.get('question', '') if extracted else ''
        
        if cat and question and cat != 'unknown':
            clean_text = preprocess_text(question, stop_words, stemmer)
            if len(clean_text.split()) > 2: # Mindestens 3 Wörter
                texts.append(clean_text)
                categories.append(cat)
                ids.append(entry.get('task_id', 'unknown'))

    print(f"ℹ️  Analysiere {len(texts)} Aufgaben.")

    # 1. TF-IDF Vektorisierung
    print("🧮 Berechne TF-IDF Vektoren...")
    vectorizer = TfidfVectorizer(max_features=1000, min_df=5, max_df=0.9)
    X = vectorizer.fit_transform(texts)
    
    # 2. K-Means Clustering
    # Wir nehmen k=4 (da wir 4 Hauptkategorien haben: Arithmetik, Algebra, Geometrie, Stochastik)
    k = 4
    print(f"🧩 Führe K-Means Clustering durch (k={k})...")
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X)
    
    # 3. Dimensionalitätsreduktion für Visualisierung (t-SNE)
    print("📉 Reduziere Dimensionen mit t-SNE (das kann einen Moment dauern)...")
    # Erst PCA auf 50 Dimensionen um Rauschen zu reduzieren und t-SNE zu beschleunigen
    pca = PCA(n_components=50)
    X_pca = pca.fit_transform(X.toarray())
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, init='pca', learning_rate='auto')
    X_embedded = tsne.fit_transform(X_pca)
    
    # 4. Visualisierung
    print("🎨 Erstelle Plots...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Plot 1: Gefundene Cluster
    sns.scatterplot(
        x=X_embedded[:, 0], y=X_embedded[:, 1],
        hue=clusters, palette='viridis', s=60, alpha=0.7, ax=ax1, legend='full'
    )
    ax1.set_title(f'Gefundene Text-Cluster (K-Means, k={k})', fontsize=14)
    ax1.set_xlabel('t-SNE Dim 1')
    ax1.set_ylabel('t-SNE Dim 2')
    
    # Top Terms pro Cluster anzeigen
    print("\n🔑 Top-Begriffe pro Cluster:")
    feature_names = vectorizer.get_feature_names_out()
    ordered_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
    cluster_labels = {}
    
    for i in range(k):
        top_terms = [feature_names[ind] for ind in ordered_centroids[i, :5]]
        print(f"Cluster {i}: {', '.join(top_terms)}")
        cluster_labels[i] = f"C{i}\n({top_terms[0]})"
        
        # Label im Plot hinzufügen (Zentroid im t-SNE Raum annähern - vereinfacht)
        # Wir nehmen einfach den Median der Punkte im Cluster für das Label
        mask = clusters == i
        if np.any(mask):
            center = np.median(X_embedded[mask], axis=0)
            ax1.text(center[0], center[1], str(i), fontsize=12, fontweight='bold', 
                     bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    # Plot 2: Wahre Kategorien
    sns.scatterplot(
        x=X_embedded[:, 0], y=X_embedded[:, 1],
        hue=categories, palette='deep', s=60, alpha=0.7, ax=ax2
    )
    ax2.set_title('Wahre Mathematische Kategorien', fontsize=14)
    ax2.set_xlabel('t-SNE Dim 1')
    ax2.set_ylabel('t-SNE Dim 2')
    
    plt.tight_layout()
    plt.savefig(output_plot, dpi=300)
    print(f"✅ Plot gespeichert unter: {output_plot}")
    
    # 5. Metriken
    print("\n📊 Cluster-Qualität (Vergleich mit Kategorien):")
    # Wir mappen die String-Kategorien auf Zahlen für den Vergleich
    unique_cats = list(set(categories))
    cat_to_id = {cat: i for i, cat in enumerate(unique_cats)}
    true_labels = [cat_to_id[c] for c in categories]
    
    h_score = homogeneity_score(true_labels, clusters)
    c_score = completeness_score(true_labels, clusters)
    v_score = v_measure_score(true_labels, clusters)
    
    print(f"Homogeneity:  {h_score:.3f} (Sind Cluster rein?)")
    print(f"Completeness: {c_score:.3f} (Sind alle Elemente einer Klasse im selben Cluster?)")
    print(f"V-Measure:    {v_score:.3f} (Harmonisches Mittel)")
    
    print("\nInterpretation:")
    print("- Hohe Homogenität bedeutet, dass ein Cluster hauptsächlich Aufgaben EINER Kategorie enthält.")
    print("- Hohe Completeness bedeutet, dass alle Aufgaben einer Kategorie im selben Cluster landen.")
    print("- Niedrige Werte deuten darauf hin, dass der Aufgabentext allein (ohne Bild) die Kategorie nicht eindeutig bestimmt.")

if __name__ == "__main__":
    main()
