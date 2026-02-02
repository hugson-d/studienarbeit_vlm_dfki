import json
import os

from datasets import Dataset, DatasetDict, Features, Image, Sequence, Value
from huggingface_hub import create_repo, delete_repo, login

# --- KONFIGURATION ---
REPO_ID = "dfkiuser/kangaroo_math_mc_questions"  # Ihr Ziel-Repo
INPUT_DATA_FILE = "dataset_final.json"  # Ihre JSON-Datei mit allen Infos
IMAGE_ROOT_DIR = "data"  # Wo der Ordner 'dataset_final' liegt (Root)
IMAGE_REPO_PREFIX = "data"  # Ordnerpfad im HF-Repo, unter dem die Bilder liegen


def build_dataset():
  """
  Erstellt ein HF-Dataset (Parquet) mit Bildspalte und Metadaten.
  """
  print(f"--- Starte Verarbeitung von {INPUT_DATA_FILE} ---")

  if not os.path.exists(INPUT_DATA_FILE):
    raise FileNotFoundError(f"Die Datei {INPUT_DATA_FILE} wurde nicht gefunden.")

  with open(INPUT_DATA_FILE, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

  rows = []
  missing_images = []

  for entry in raw_data:
    image_rel_path = entry.get("image_path")
    if not image_rel_path:
      missing_images.append("<missing image_path>")
      continue

    full_path = os.path.join(IMAGE_ROOT_DIR, image_rel_path)
    if not os.path.exists(full_path):
      missing_images.append(image_rel_path)
      continue

    rows.append(
      {
        "image": os.path.abspath(full_path),
        "question": entry["extracted_text"]["question"],
        "choices": entry["extracted_text"]["answer_options"],
        "answer": entry["answer"],
        "year": int(entry["year"]),
        "class": entry["class"],
        "task_id": entry["task_id"],
        "math_category": entry["math_category"],
      }
    )

  print("--- Abschlussbericht ---")
  print(f"Valide Einträge: {len(rows)}")
  if missing_images:
    print(f"Fehlende Bilder ({len(missing_images)}): {missing_images[:3]} ...")
  else:
    print("Alle Bildpfade sind korrekt.")

  features = Features(
    {
      "image": Image(),
      "question": Value("string"),
      "choices": Sequence(Value("string")),
      "answer": Value("string"),
      "year": Value("int32"),
      "class": Value("string"),
      "task_id": Value("string"),
      "math_category": Value("string"),
    }
  )

  dataset = Dataset.from_list(rows, features=features)
  return DatasetDict({"train": dataset})


def create_readme():
    """Erstellt eine minimale Dataset Card."""
    readme_content = """---
readme: "generated_by_script"
language:
  - de
task_categories:
  - visual-question-answering
tags:
  - math
  - education
---

# Kangaroo Math Dataset

Dieses Dataset wurde automatisch hochgeladen. Es folgt der Struktur von VLM-Benchmarks.

## Struktur
- **image**: Das Bild zur Aufgabe
- **question**: Die Textfrage
- **choices**: Antwortmöglichkeiten
- **answer**: Die korrekte Antwort (Label)
"""
    with open("README.md", "w", encoding="utf-8") as f:
        f.write(readme_content)
    print("README.md erstellt.")


def main():
    # 1. Login (fragt nach Token, wenn nicht gecached)
    # Wenn Sie das Token im Code hardcoden wollen (nicht empfohlen): login(token="hf_...")
    login()

    # 2. Repo neu anlegen (vorherige Version entfernen)
    delete_repo(repo_id=REPO_ID, repo_type="dataset", missing_ok=True)
    create_repo(repo_id=REPO_ID, repo_type="dataset", exist_ok=True)

    # 3. Daten vorbereiten
    dataset_dict = build_dataset()

    # 4. Readme erstellen (wichtig für die korrekte Anzeige im Browser)
    create_readme()

    # 5. Upload (Parquet/Arrow via datasets)
    print(f"--- Starte Upload zu {REPO_ID} ---")
    dataset_dict.push_to_hub(
      REPO_ID,
      commit_message="Upload dataset as parquet",
    )
    print("Upload erfolgreich! Das Dataset ist nun live und 'load_dataset'-bereit.")


if __name__ == "__main__":
    main()
