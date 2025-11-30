# Analyse des Känguru-Mathematik-Datasets
*Generiert am 28.11.2025*
## 1. Einleitung und Datensatzbeschreibung
Der vorliegende Datensatz umfasst insgesamt **3,557** Mathematikaufgaben aus dem Känguru-Wettbewerb der Jahre **1998 bis 2025**. Der Datensatz wurde für die Evaluation von Vision-Language-Models (VLMs) kuratiert und enthält sowohl die visuellen Repräsentationen (Bilder) als auch semantische Metadaten.
## 2. Temporale und Strukturelle Verteilung
### 2.1 Zeitliche Abdeckung
Die Verteilung der Aufgaben über den Erfassungszeitraum ist wie folgt:
| Jahr | Anzahl Aufgaben | Anteil |
|------|-----------------|--------|
| 1998 | 102 | 2.9% |
| 1999 | 133 | 3.7% |
| 2000 | 135 | 3.8% |
| 2001 | 107 | 3.0% |
| 2002 | 129 | 3.6% |
| 2003 | 133 | 3.7% |
| 2004 | 137 | 3.9% |
| 2005 | 125 | 3.5% |
| 2006 | 130 | 3.7% |
| 2007 | 130 | 3.7% |
| 2008 | 107 | 3.0% |
| 2009 | 129 | 3.6% |
| 2010 | 125 | 3.5% |
| 2011 | 134 | 3.8% |
| 2012 | 131 | 3.7% |
| 2013 | 131 | 3.7% |
| 2014 | 134 | 3.8% |
| 2015 | 135 | 3.8% |
| 2016 | 132 | 3.7% |
| 2017 | 131 | 3.7% |
| 2018 | 127 | 3.6% |
| 2019 | 122 | 3.4% |
| 2020 | 123 | 3.5% |
| 2021 | 122 | 3.4% |
| 2022 | 126 | 3.5% |
| 2023 | 131 | 3.7% |
| 2024 | 124 | 3.5% |
| 2025 | 132 | 3.7% |

### 2.2 Verteilung nach Klassenstufen
Der Datensatz deckt alle relevanten Altersgruppen des Wettbewerbs ab:
| Klassenstufe | Anzahl Aufgaben | Anteil |
|--------------|-----------------|--------|
| 3./4. Klasse | 584 | 16.4% |
| 5./6. Klasse | 690 | 19.4% |
| 7./8. Klasse | 783 | 22.0% |
| 9./10. Klasse | 777 | 21.8% |
| 11.-13. Klasse | 723 | 20.3% |

### 2.3 Komplexitätsverteilung (Schwierigkeitsgrad)
Die Aufgaben sind in drei Schwierigkeitsgrade unterteilt (A: 3 Punkte, B: 4 Punkte, C: 5 Punkte):
| Schwierigkeitsgrad | Anzahl Aufgaben | Anteil |
|--------------------|-----------------|--------|
| A (Leicht / 3 Pkt) | 1172 | 32.9% |
| B (Mittel / 4 Pkt) | 1184 | 33.3% |
| C (Schwer / 5 Pkt) | 1201 | 33.8% |

## 3. Inhaltliche Analyse (Mathematische Teilgebiete)
Die Aufgaben wurden mittels GPT-4o Vision in vier mathematische Hauptkategorien klassifiziert:
| Kategorie | Anzahl Aufgaben | Anteil |
|-----------|-----------------|--------|
| Geometrie | 1168 | 32.8% |
| Arithmetik | 1109 | 31.2% |
| Algebra | 593 | 16.7% |
| Stochastik | 593 | 16.7% |
| unknown | 94 | 2.6% |

### 3.1 Themenverteilung nach Klassenstufe
Die folgende Matrix zeigt die thematischen Schwerpunkte je Altersgruppe:
| Klassenstufe | Geometrie | Arithmetik | Algebra | Stochastik |
|--------------|---|---|---|---|
| 3./4. Klasse | 189 | 235 | 56 | 85 |
| 5./6. Klasse | 235 | 251 | 75 | 110 |
| 7./8. Klasse | 254 | 270 | 107 | 131 |
| 9./10. Klasse | 258 | 221 | 142 | 137 |
| 11.-13. Klasse | 232 | 132 | 213 | 130 |

## 4. Modalitätsanalyse (Text vs. Visuell)
Ein zentraler Aspekt für die VLM-Evaluation ist die Unterscheidung zwischen Aufgaben, die rein textbasiert lösbar sind, und solchen, die visuelle Elemente erfordern.
- **Text-Only Aufgaben**: 1572 (44.2% der analysierten Aufgaben)
- **Visuell-Notwendige Aufgaben**: 1985 (55.8% der analysierten Aufgaben)

### 4.1 Visuelle Abhängigkeit nach Kategorie
| Kategorie | Text-Only | Visuell | Visueller Anteil |
|-----------|-----------|---------|------------------|
| Geometrie | 200 | 968 | **82.9%** |
| Arithmetik | 681 | 428 | **38.6%** |
| Algebra | 350 | 243 | **41.0%** |
| Stochastik | 308 | 285 | **48.1%** |

### 4.2 Visuelle Aufgaben: Schwierigkeit und Antwortverteilung
Um sicherzustellen, dass visuelle Aufgaben nicht systematisch schwerer oder leichter sind, wird hier die Verteilung für die Teilmenge der visuell-notwendigen Aufgaben betrachtet.
#### Schwierigkeitsgrad (Visuell)
| Schwierigkeitsgrad | Anzahl (Visuell) | Anteil |
|--------------------|------------------|--------|
| A (Leicht / 3 Pkt) | 658 | 33.1% |
| B (Mittel / 4 Pkt) | 646 | 32.5% |
| C (Schwer / 5 Pkt) | 681 | 34.3% |

#### Antwortverteilung (Visuell)
| Antwort | Anzahl (Visuell) | Anteil |
|---------|------------------|--------|
| A | 371 | 18.7% |
| B | 414 | 20.9% |
| C | 404 | 20.4% |
| D | 422 | 21.3% |
| E | 373 | 18.8% |

#### Mathematische Kategorien (Visuell)
| Kategorie | Anzahl (Visuell) | Anteil |
|-----------|------------------|--------|
| Geometrie | 968 | 50.3% |
| Arithmetik | 428 | 22.2% |
| Algebra | 243 | 12.6% |
| Stochastik | 285 | 14.8% |

## 5. Textuelle Eigenschaften
- **Durchschnittliche Wortanzahl pro Frage**: 37.3 Wörter
- **Median der Wortanzahl**: 35.0 Wörter
- **Min/Max**: 3 / 113 Wörter

## 6. Antwortverteilung (Bias-Check)
Um sicherzustellen, dass das Modell nicht durch Raten einer häufigsten Antwort (z.B. immer 'C') eine künstlich hohe Performance erzielt, wurde die Verteilung der korrekten Antwortbuchstaben analysiert.
| Antwort | Anzahl | Anteil |
|---------|--------|--------|
| A | 646 | 18.2% |
| B | 733 | 20.6% |
| C | 739 | 20.8% |
| D | 773 | 21.7% |
| E | 664 | 18.7% |

## 7. Linguistische Analyse
### 7.1 Methodik
Für die Analyse der häufigsten Begriffe wurden folgende Vorverarbeitungsschritte durchgeführt:
1. **Normalisierung**: Umwandlung in Kleinschreibung.
2. **Bereinigung**: Entfernung von Satzzeichen und Zahlen.
3. **Stopword-Removal**: Entfernung häufiger deutscher Füllwörter (basierend auf NLTK 'german' corpus) sowie domänenspezifischer Begriffe (z.B. 'dass', 'wurde').
4. **Filterung**: Ausschluss von Wörtern mit weniger als 3 Buchstaben.

### 7.2 Häufigste Begriffe (Top 25)
| Rang | Begriff | Häufigkeit |
|------|---------|------------|
| 1 | viele | 1177 |
| 2 | zahlen | 974 |
| 3 | zahl | 703 |
| 4 | zwei | 473 |
| 5 | drei | 449 |
| 6 | summe | 384 |
| 7 | beiden | 350 |
| 8 | genau | 349 |
| 9 | gleich | 340 |
| 10 | vier | 303 |
| 11 | würfel | 291 |
| 12 | rechts | 289 |
| 13 | ziffern | 289 |
| 14 | groß | 284 |
| 15 | folgenden | 274 |
| 16 | punkte | 257 |
| 17 | verschiedene | 211 |
| 18 | flächeninhalt | 210 |
| 19 | insgesamt | 207 |
| 20 | bild | 197 |
| 21 | lang | 196 |
| 22 | abgebildeten | 190 |
| 23 | quadrat | 182 |
| 24 | mindestens | 182 |
| 25 | fünf | 180 |

### 7.3 WordCloud
![WordCloud der häufigsten Begriffe](wordcloud.png)

## 8. Vollständigkeitsanalyse
Der Datensatz wurde gegen die offiziellen Lösungsschlüssel abgeglichen, um die Vollständigkeit zu verifizieren.
- **Erwartete Aufgaben**: 3888
- **Gefundene Aufgaben**: 3557
- **Abdeckung (Coverage)**: 91.5%

### 8.1 Fehlende Aufgaben (331)
Folgende Aufgaben sind im Lösungsschlüssel verzeichnet, konnten aber nicht extrahiert werden (z.B. aufgrund von PDF-Fehlern oder fehlenden Seiten):
1998_11bis13_1, 1998_11bis13_15, 1998_11bis13_3, 1998_11bis13_4, 1998_11bis13_5, 1998_11bis13_6, 1998_11bis13_7, 1998_11bis13_8, 1998_11bis13_9, 1998_3und4_11 ... 2024_9und10_B4, 2024_9und10_C1, 2024_9und10_C2, 2024_9und10_C9, 2025_11bis13_A6, 2025_11bis13_B9, 2025_5und6_B5, 2025_7und8_B10, 2025_7und8_B8, 2025_9und10_C8

### 8.2 Detaillierte Abdeckung nach Jahr und Klasse
Die folgende Tabelle zeigt die Anzahl der extrahierten Aufgaben im Vergleich zu den erwarteten Aufgaben für jede Jahrgangsstufe und jedes Jahr.
| Jahr | Klasse | Gefunden | Erwartet | Fehlend | Quote |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1998** | 3und4 | 7 | 15 | **8** | 46.7% |
| **1998** | 5und6 | 17 | 30 | **13** | 56.7% |
| **1998** | 7und8 | 28 | 30 | **2** | 93.3% |
| **1998** | 9und10 | 29 | 30 | **1** | 96.7% |
| **1998** | 11bis13 | 21 | 30 | **9** | 70.0% |
| 1999 | 3und4 | 15 | 15 | 0 | 100.0% |
| **1999** | 5und6 | 29 | 30 | **1** | 96.7% |
| 1999 | 7und8 | 30 | 30 | 0 | 100.0% |
| 1999 | 9und10 | 30 | 30 | 0 | 100.0% |
| **1999** | 11bis13 | 29 | 30 | **1** | 96.7% |
| **2000** | 3und4 | 20 | 21 | **1** | 95.2% |
| 2000 | 5und6 | 30 | 30 | 0 | 100.0% |
| **2000** | 7und8 | 28 | 30 | **2** | 93.3% |
| **2000** | 9und10 | 29 | 30 | **1** | 96.7% |
| **2000** | 11bis13 | 28 | 30 | **2** | 93.3% |
| 2001 | 3und4 | 21 | 21 | 0 | 100.0% |
| **2001** | 5und6 | 29 | 30 | **1** | 96.7% |
| 2001 | 7und8 | 30 | 30 | 0 | 100.0% |
| **2001** | 9und10 | 27 | 30 | **3** | 90.0% |
| **2001** | **11bis13** | 0 | **30** | **30** | **0.0%** |
| 2002 | 3und4 | 21 | 21 | 0 | 100.0% |
| 2002 | 5und6 | 30 | 30 | 0 | 100.0% |
| **2002** | 7und8 | 26 | 30 | **4** | 86.7% |
| **2002** | 9und10 | 27 | 30 | **3** | 90.0% |
| **2002** | 11bis13 | 25 | 30 | **5** | 83.3% |
| 2003 | 3und4 | 21 | 21 | 0 | 100.0% |
| **2003** | 5und6 | 26 | 30 | **4** | 86.7% |
| **2003** | 7und8 | 29 | 30 | **1** | 96.7% |
| **2003** | 9und10 | 28 | 30 | **2** | 93.3% |
| **2003** | 11bis13 | 29 | 30 | **1** | 96.7% |
| 2004 | 3und4 | 21 | 21 | 0 | 100.0% |
| 2004 | 5und6 | 30 | 30 | 0 | 100.0% |
| **2004** | 7und8 | 28 | 30 | **2** | 93.3% |
| **2004** | 9und10 | 29 | 30 | **1** | 96.7% |
| **2004** | 11bis13 | 29 | 30 | **1** | 96.7% |
| **2005** | 3und4 | 20 | 21 | **1** | 95.2% |
| **2005** | 5und6 | 26 | 30 | **4** | 86.7% |
| **2005** | 7und8 | 25 | 30 | **5** | 83.3% |
| **2005** | 9und10 | 26 | 30 | **4** | 86.7% |
| **2005** | 11bis13 | 28 | 30 | **2** | 93.3% |
| **2006** | 3und4 | 20 | 21 | **1** | 95.2% |
| **2006** | 5und6 | 25 | 30 | **5** | 83.3% |
| **2006** | 7und8 | 29 | 30 | **1** | 96.7% |
| **2006** | 9und10 | 29 | 30 | **1** | 96.7% |
| **2006** | 11bis13 | 27 | 30 | **3** | 90.0% |
| 2007 | 3und4 | 21 | 21 | 0 | 100.0% |
| 2007 | 5und6 | 30 | 30 | 0 | 100.0% |
| **2007** | 7und8 | 24 | 30 | **6** | 80.0% |
| **2007** | 9und10 | 27 | 30 | **3** | 90.0% |
| **2007** | 11bis13 | 28 | 30 | **2** | 93.3% |
| **2008** | 3und4 | 20 | 21 | **1** | 95.2% |
| 2008 | 5und6 | 30 | 30 | 0 | 100.0% |
| **2008** | 7und8 | 28 | 30 | **2** | 93.3% |
| **2008** | 9und10 | 29 | 30 | **1** | 96.7% |
| **2008** | **11bis13** | 0 | **30** | **30** | **0.0%** |
| 2009 | 3und4 | 21 | 21 | 0 | 100.0% |
| **2009** | 5und6 | 28 | 30 | **2** | 93.3% |
| **2009** | 7und8 | 25 | 30 | **5** | 83.3% |
| **2009** | 9und10 | 25 | 30 | **5** | 83.3% |
| 2009 | 11bis13 | 30 | 30 | 0 | 100.0% |
| **2010** | 3und4 | 23 | 24 | **1** | 95.8% |
| **2010** | 5und6 | 21 | 24 | **3** | 87.5% |
| **2010** | 7und8 | 26 | 30 | **4** | 86.7% |
| **2010** | 9und10 | 28 | 30 | **2** | 93.3% |
| **2010** | 11bis13 | 27 | 30 | **3** | 90.0% |
| **2011** | 3und4 | 23 | 24 | **1** | 95.8% |
| 2011 | 5und6 | 24 | 24 | 0 | 100.0% |
| **2011** | 7und8 | 29 | 30 | **1** | 96.7% |
| **2011** | 9und10 | 29 | 30 | **1** | 96.7% |
| **2011** | 11bis13 | 29 | 30 | **1** | 96.7% |
| 2012 | 3und4 | 24 | 24 | 0 | 100.0% |
| **2012** | 5und6 | 23 | 24 | **1** | 95.8% |
| **2012** | 7und8 | 29 | 30 | **1** | 96.7% |
| **2012** | 9und10 | 28 | 30 | **2** | 93.3% |
| **2012** | 11bis13 | 27 | 30 | **3** | 90.0% |
| 2013 | 3und4 | 24 | 24 | 0 | 100.0% |
| **2013** | 5und6 | 22 | 24 | **2** | 91.7% |
| **2013** | 7und8 | 29 | 30 | **1** | 96.7% |
| **2013** | 9und10 | 27 | 30 | **3** | 90.0% |
| **2013** | 11bis13 | 29 | 30 | **1** | 96.7% |
| **2014** | 3und4 | 22 | 24 | **2** | 91.7% |
| 2014 | 5und6 | 24 | 24 | 0 | 100.0% |
| **2014** | 7und8 | 29 | 30 | **1** | 96.7% |
| 2014 | 9und10 | 30 | 30 | 0 | 100.0% |
| **2014** | 11bis13 | 29 | 30 | **1** | 96.7% |
| 2015 | 3und4 | 24 | 24 | 0 | 100.0% |
| 2015 | 5und6 | 24 | 24 | 0 | 100.0% |
| **2015** | 7und8 | 29 | 30 | **1** | 96.7% |
| **2015** | 9und10 | 29 | 30 | **1** | 96.7% |
| **2015** | 11bis13 | 29 | 30 | **1** | 96.7% |
| **2016** | 3und4 | 23 | 24 | **1** | 95.8% |
| **2016** | 5und6 | 23 | 24 | **1** | 95.8% |
| **2016** | 7und8 | 28 | 30 | **2** | 93.3% |
| **2016** | 9und10 | 28 | 30 | **2** | 93.3% |
| 2016 | 11bis13 | 30 | 30 | 0 | 100.0% |
| **2017** | 3und4 | 20 | 24 | **4** | 83.3% |
| 2017 | 5und6 | 24 | 24 | 0 | 100.0% |
| **2017** | 7und8 | 29 | 30 | **1** | 96.7% |
| **2017** | 9und10 | 29 | 30 | **1** | 96.7% |
| **2017** | 11bis13 | 29 | 30 | **1** | 96.7% |
| **2018** | 3und4 | 21 | 24 | **3** | 87.5% |
| **2018** | 5und6 | 22 | 24 | **2** | 91.7% |
| **2018** | 7und8 | 28 | 30 | **2** | 93.3% |
| **2018** | 9und10 | 27 | 30 | **3** | 90.0% |
| **2018** | 11bis13 | 29 | 30 | **1** | 96.7% |
| **2019** | 3und4 | 19 | 24 | **5** | 79.2% |
| **2019** | 5und6 | 22 | 24 | **2** | 91.7% |
| **2019** | 7und8 | 28 | 30 | **2** | 93.3% |
| **2019** | 9und10 | 29 | 30 | **1** | 96.7% |
| **2019** | 11bis13 | 24 | 30 | **6** | 80.0% |
| **2020** | 3und4 | 22 | 24 | **2** | 91.7% |
| **2020** | 5und6 | 21 | 24 | **3** | 87.5% |
| **2020** | 7und8 | 28 | 30 | **2** | 93.3% |
| **2020** | 9und10 | 25 | 30 | **5** | 83.3% |
| **2020** | 11bis13 | 27 | 30 | **3** | 90.0% |
| **2021** | 3und4 | 21 | 24 | **3** | 87.5% |
| **2021** | 5und6 | 21 | 24 | **3** | 87.5% |
| **2021** | 7und8 | 27 | 30 | **3** | 90.0% |
| **2021** | 9und10 | 25 | 30 | **5** | 83.3% |
| **2021** | 11bis13 | 28 | 30 | **2** | 93.3% |
| **2022** | 3und4 | 22 | 24 | **2** | 91.7% |
| **2022** | 5und6 | 21 | 24 | **3** | 87.5% |
| **2022** | 7und8 | 28 | 30 | **2** | 93.3% |
| **2022** | 9und10 | 27 | 30 | **3** | 90.0% |
| **2022** | 11bis13 | 28 | 30 | **2** | 93.3% |
| **2023** | 3und4 | 23 | 24 | **1** | 95.8% |
| **2023** | 5und6 | 23 | 24 | **1** | 95.8% |
| 2023 | 7und8 | 30 | 30 | 0 | 100.0% |
| **2023** | 9und10 | 27 | 30 | **3** | 90.0% |
| **2023** | 11bis13 | 28 | 30 | **2** | 93.3% |
| **2024** | 3und4 | 21 | 24 | **3** | 87.5% |
| **2024** | 5und6 | 22 | 24 | **2** | 91.7% |
| **2024** | 7und8 | 28 | 30 | **2** | 93.3% |
| **2024** | 9und10 | 25 | 30 | **5** | 83.3% |
| **2024** | 11bis13 | 28 | 30 | **2** | 93.3% |
| 2025 | 3und4 | 24 | 24 | 0 | 100.0% |
| **2025** | 5und6 | 23 | 24 | **1** | 95.8% |
| **2025** | 7und8 | 28 | 30 | **2** | 93.3% |
| **2025** | 9und10 | 29 | 30 | **1** | 96.7% |
| **2025** | 11bis13 | 28 | 30 | **2** | 93.3% |

---
Diese Analyse dient als Grundlage für die differenzierte Auswertung der Modell-Leistungen in der Studienarbeit.