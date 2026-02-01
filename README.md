---
readme: "generated_by_script"
dataset_info:
  features:
    - name: image
      dtype: image
    - name: question
      dtype: string
    - name: choices
      sequence: string
    - name: answer
      dtype: string
    - name: year
      dtype: int32
    - name: class
      dtype: string
    - name: math_category
      dtype: string
    - name: task_id
      dtype: string
configs:
  - config_name: default
    data_files: "metadata.jsonl"
---

# Kangaroo Math Dataset

Dieses Dataset wurde automatisch hochgeladen. Es folgt der Struktur von VLM-Benchmarks.

## Struktur
- **image**: Das Bild zur Aufgabe
- **question**: Die Textfrage
- **choices**: Antwortmöglichkeiten
- **answer**: Die korrekte Antwort (Label)
