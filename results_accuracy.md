# VLM Benchmark Ergebnisse: Accuracy Analyse

Hier sind die berechneten Accuracy-Ergebnisse für alle JSONL-Dateien im `results_vllm` Ordner als Markdown-Tabelle. Die Tabelle zeigt die Gesamtaccuracy pro Modell sowie die Accuracy pro `math_category`. Alle Modelle haben **0 ungültige Formate** (`format_valid != true`).

## Übersicht
- **Modelle getestet:** 16 (Gemma-3 Varianten, Idefics3, InternVL3 Varianten, Ovis2.5 Varianten, Qwen2.5-VL Varianten, inkl. CoT-Voting Varianten)
- **Gesamtanzahl Tasks pro Modell:** 3557
- **Accuracy-Bereich:** 19.9% bis 54.4%

## Accuracy-Tabelle

| Modell                  | Gesamt Accuracy | Algebra | Arithmetik | Geometrie | Stochastik | unknown | None  |
|-------------------------|-----------------|---------|------------|-----------|------------|---------|-------|
| Gemma-3-12B-vLLM       | 23.0% (819/3557) | 24.5% | 21.5% | 23.5% | 22.4% | 29.7% | 33.3% |
| Gemma-3-27B-vLLM       | 22.2% (788/3557) | 21.4% | 21.7% | 22.7% | 22.9% | 20.9% | 0.0% |
| Gemma-3-4B-vLLM        | 21.2% (754/3557) | 21.2% | 19.1% | 21.2% | 25.5% | 18.7% | 0.0% |
| Idefics3-8B-Llama3-vLLM| 19.9% (708/3557) | 21.1% | 19.5% | 20.8% | 18.5% | 14.3% | 33.3% |
| InternVL3-14B-vLLM     | 27.5% (977/3557) | 26.5% | 28.5% | 26.5% | 27.3% | 35.2% | 0.0% |
| InternVL3-38B-vLLM     | 31.3% (1112/3557)| 31.7% | 31.2% | 29.1% | 35.1% | 33.0% | 0.0% |
| InternVL3-8B-vLLM      | 22.7% (807/3557) | 21.6% | 22.7% | 22.9% | 22.6% | 28.6% | 0.0% |
| Ovis2.5-2B-CoT-Voting-n5| 34.5% (1226/3557)| 39.3% | 41.7% | 28.2% | 29.2% | 30.8% | 0.0% |
| Ovis2.5-2B-vLLM        | 25.4% (904/3557) | 24.6% | 23.4% | 28.1% | 24.5% | 28.6% | 0.0% |
| Ovis2.5-9B-CoT-Voting-n5| 54.4% (1934/3557)| 64.4% | 67.5% | 41.0% | 47.9% | 44.0% | 0.0% |
| Ovis2.5-9B-vLLM        | 31.1% (1105/3557)| 31.1% | 30.8% | 29.5% | 32.0% | 36.3% | 0.0% |
| Qwen2.5-VL-32B-vLLM    | 37.9% (1347/3557)| 39.6% | 39.9% | 34.2% | 38.8% | 42.9% | 0.0% |
| Qwen2.5-VL-3B-Instruct-CoT-Voting-n5| 29.3% (1043/3557)| 31.9% | 34.9% | 25.2% | 25.1% | 26.4% | 0.0% |
| Qwen2.5-VL-3B-vLLM     | 25.2% (898/3557) | 25.6% | 25.5% | 24.6% | 25.6% | 26.4% | 0.0% |
| Qwen2.5-VL-72B-AWQ-vLLM| 40.8% (1453/3557)| 42.8% | 45.2% | 35.0% | 43.3% | 34.1% | 33.3% |
| Qwen2.5-VL-7B-vLLM     | 28.6% (1016/3557)| 29.0% | 28.7% | 27.0% | 31.5% | 25.3% | 33.3% |

## Hinweise
- **Format:** Alle Werte sind in Prozent mit absoluten Zahlen (korrekt/total).
- **Kategorien:** "None" tritt selten auf (meist 0/3 Tasks) und scheint ein Datenfehler zu sein. "unknown" hat 91 Tasks.
- **Beste Modelle:** Ovis2.5-9B-CoT-Voting-n5 (54.4%), Qwen2.5-VL-72B-AWQ (40.8%).
- **CoT-Voting:** Die CoT-Voting-Varianten zeigen deutlich höhere Accuracy, besonders Ovis2.5-9B mit 54.4%.

Falls du weitere Analysen (z.B. Plots) möchtest, lass es mich wissen! Das Analyse-Skript liegt in `src/analyze_accuracy.py`.