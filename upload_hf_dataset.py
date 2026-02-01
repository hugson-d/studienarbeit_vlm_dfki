import json
import os
from huggingface_hub import create_repo, login, upload_folder

# --- KONFIGURATION ---
REPO_ID = "hugsonszilla/kangaroo_math"  # Ihr Ziel-Repo
INPUT_DATA_FILE = "dataset_final.json"  # Ihre JSON-Datei mit allen Infos
IMAGE_ROOT_DIR = "data"  # Wo der Ordner 'dataset_final' liegt (Root)
IMAGE_REPO_PREFIX = "data"  # Ordnerpfad im HF-Repo, unter dem die Bilder liegen
METADATA_FILE = "metadata.jsonl"  # Die Datei, die wir generieren (HF Standard)


def create_metadata_and_validate():
    """
    Konvertiert die geschachtelte JSON in das flache HF-Format (metadata.jsonl)
    und validiert die Bildpfade.
    """
    print(f"--- Starte Verarbeitung von {INPUT_DATA_FILE} ---")

    if not os.path.exists(INPUT_DATA_FILE):
        raise FileNotFoundError(f"Die Datei {INPUT_DATA_FILE} wurde nicht gefunden.")

    with open(INPUT_DATA_FILE, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    valid_entries = 0
    missing_images = []

    # Wir schreiben Zeile für Zeile in die metadata.jsonl
    with open(METADATA_FILE, "w", encoding="utf-8") as outfile:
        for entry in raw_data:
            # 1. Pfad-Prüfung
            # Der Pfad im JSON ist z.B. "dataset_final/2025_...png"
            image_rel_path = entry.get("image_path")
            if not image_rel_path:
                missing_images.append("<missing image_path>")
                continue

            full_path = os.path.join(IMAGE_ROOT_DIR, image_rel_path)

            if not os.path.exists(full_path):
                print(f"WARNUNG: Bild nicht gefunden: {full_path}")
                missing_images.append(image_rel_path)
                continue  # Diesen Eintrag überspringen, um Dataset nicht zu korrumpieren

            # 2. Daten-Flattening (Flachklopfen der Struktur)
            # HF erwartet 'file_name' für das Bild mapping
            repo_image_path = (
              f"{IMAGE_REPO_PREFIX}/{image_rel_path}"
              if IMAGE_REPO_PREFIX
              else image_rel_path
            )
            hf_entry = {
              "file_name": repo_image_path,
                "question": entry["extracted_text"]["question"],
                "choices": entry["extracted_text"]["answer_options"],
                "answer": entry["answer"],
                "year": entry["year"],
                "class": entry["class"],
                "task_id": entry["task_id"],
                "math_category": entry["math_category"],
            }

            # Schreiben als JSON-Zeile
            json.dump(hf_entry, outfile, ensure_ascii=False)
            outfile.write("\n")
            valid_entries += 1

    print("--- Abschlussbericht ---")
    print(f"Valide Einträge geschrieben: {valid_entries}")
    if missing_images:
        print(f"Fehlende Bilder ({len(missing_images)}): {missing_images[:3]} ...")
    else:
        print("Alle Bildpfade sind korrekt.")


def create_readme():
    """Erstellt eine minimale Dataset Card, damit HF weiß, wie es rendern soll."""
    readme_content = """---
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
"""
    with open("README.md", "w", encoding="utf-8") as f:
        f.write(readme_content)
    print("README.md erstellt.")


def main():
    # 1. Login (fragt nach Token, wenn nicht gecached)
    # Wenn Sie das Token im Code hardcoden wollen (nicht empfohlen): login(token="hf_...")
    login()

    # 2. Repo sicherstellen (falls noch nicht vorhanden)
    create_repo(repo_id=REPO_ID, repo_type="dataset", exist_ok=True)

    # 3. Daten vorbereiten
    create_metadata_and_validate()

    # 4. Readme erstellen (wichtig für die korrekte Anzeige im Browser)
    create_readme()

    # 5. Upload
    print(f"--- Starte Upload zu {REPO_ID} ---")
    upload_folder(
        folder_path=".",
        repo_id=REPO_ID,
        repo_type="dataset",
        # WICHTIG: Wir laden nur das hoch, was nötig ist, um Müll zu vermeiden
        allow_patterns=[
        "data/dataset_final/**",  # Ihre Bilder
            "metadata.jsonl",  # Die Map
            "README.md",  # Die Doku
        ],
    )
    print("Upload erfolgreich! Das Dataset ist nun live und 'load_dataset'-bereit.")


if __name__ == "__main__":
    main()
