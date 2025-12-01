# Känguru-Wettbewerb VLM Dataset

Structured dataset and tooling for evaluating Vision Language Models (VLMs) on German Känguru math competition tasks (1998-2025).

## 📊 Dataset Overview

- **3,557 tasks** ready for VLM evaluation (`dataset_final/`)
- **235 excluded tasks** (visual/quality issues) in `dataset_final_not_used/`
- **28 years** of competition data (1998-2025)
- **5 grade levels**: 3-4, 5-6, 7-8, 9-10, 11-13
- **3 difficulty levels**: A (easy), B (medium), C (hard) - balanced at ~33% each

See [DATASET_STATS.md](DATASET_STATS.md) for detailed statistics (auto-generated).

## 🗂️ Dataset Structure

```json
{
  "image_path": "dataset_final/2024_7und8_B5.png",
  "year": 2024,
  "class": "7und8",
  "task_id": "B5",
  "answer": "C",
  "math_category": "Geometrie",
  "is_text_only": false,
  "extracted_text": {
    "question": "...",
    "answer_options": ["(A) ...", "(B) ...", ...]
  }
}
```

## 🚀 Prerequisites

- [uv](https://docs.astral.sh/uv/) (>=0.5) for Python dependency management
- Python 3.11+
- OpenAI API key (for categorization and text extraction)

Install uv on macOS:
```bash
brew install uv
```

## ⚙️ Setup

1. **Install dependencies:**
   ```bash
   uv sync
   ```

2. **Configure OpenAI API key:**
   ```bash
   cp .env.example .env
   # Edit .env and add your OPENAI_API_KEY
   ```

3. **Activate environment** (optional):
   ```bash
   source .venv/bin/activate
   ```

## 🔧 Available Scripts

### Dataset Analysis

**Analyze dataset distribution** (generates DATASET_STATS.md):
```bash
uv run python src/analyze_dataset_distribution.py
```

Shows distribution by year, class, difficulty and generates markdown statistics.

### Utility Scripts

**Create main dataset JSON:**
```bash
uv run python src/create_dataset_json.py
```
Combines task images with solutions and generates `dataset_final.json`.

**Extract tasks from PDFs:**
```bash
# For 2012-2025 (direct extraction)
uv run python src/extract_tasks_2012_2025.py

# For 1998-2011 (OCR-based extraction)
uv run python src/extract_tasks_1998_2011.py
```

## 📋 Task ID Format

### 2012-2025 (ABC Format)
- Files: `YYYY_class_A1.png` to `YYYY_class_C10.png`
- Task IDs: `A1-A10`, `B1-B10`, `C1-C10` (or A1-A8, B1-B8, C1-C8 for grades 3-4, 5-6)

### 1998-2011 (Converted to ABC)
- Files: `YYYY_class_1.png` to `YYYY_class_30.png` (numeric)
- Task IDs: **Converted** to ABC format in JSON for consistency

See [MAPPING_LOGIC.md](MAPPING_LOGIC.md) for detailed conversion rules.

## 📝 Difficulty Mapping (1998-2011)

**Grades 3-4 and 5-6:**
- A (Easy): Tasks 1-8 → A1-A8
- B (Medium): Tasks 9-16 → B1-B8
- C (Hard): Tasks 17-24 → C1-C8

**Grades 7-8, 9-10, 11-13:**
- A (Easy): Tasks 1-10 → A1-A10
- B (Medium): Tasks 11-20 → B1-B10
- C (Hard): Tasks 21-30 → C1-C10

**Note:** 1998 uses "Punkte-Fragen" format instead of "Punkte-Aufgaben" (different terminology).

## 🔄 Data Extraction

The dataset was built using multiple extraction methods:

1. **2012-2025:** Direct PDF extraction with PyMuPDF
2. **1998-2011:** OCR-based extraction (PDFs have encoding issues)
   - Uses Tesseract OCR with German language support
   - Marker detection: "3-Punkte-Fragen" (1998) or "3-Punkte-Aufgaben" (2000+)
   - Special case: 1998 Grade 3-4 starts with "6-Punkte-Fragen"

## 🗃️ Repository Structure

```
├── dataset_final.json          # Main dataset (3557 tasks)
├── DATASET_STATS.md            # Auto-generated statistics
├── MAPPING_LOGIC.md            # Task ID conversion documentation
├── data/
│   ├── dataset_final/          # Task images (for evaluation)
│   ├── dataset_final_not_used/ # Excluded task images (235)
│   ├── kanguru_pdfs/           # Processed PDF files (1998-2009)
│   ├── lösungen_1998_2011.json # Solutions 1998-2011 (sorted by year)
│   └── lösungen_2012_2025.json # Solutions 2012-2025
└── src/
    ├── analyze_dataset_distribution.py
    ├── categorize_math_tasks.py
    ├── analyze_text_only.py
    ├── extract_text.py
    ├── extract_tasks_1998_2011.py  # OCR-based extraction
    ├── extract_tasks_2012_2025.py  # Direct PDF extraction
    ├── create_dataset_json.py      # Dataset builder
    └── create_solutions_*.py       # Solution file generators
```

## 🎯 Dataset Quality

- **93.8%** usable rate (3,557 out of 3,792 extracted tasks)
- Quality filtering removes tasks with:
  - Visual artifacts or poor scan quality
  - Complex multi-page layouts
  - OCR detection failures
- See detailed statistics in [DATASET_STATS.md](DATASET_STATS.md)

## 📚 Documentation

- [DATASET_STATS.md](DATASET_STATS.md) - Detailed statistics and distributions
- [MAPPING_LOGIC.md](MAPPING_LOGIC.md) - Task ID format and conversion rules

## 🤖 VLM Evaluation

This dataset is designed for evaluating Vision Language Models on:
- Mathematical reasoning across 28 years (1998-2025)
- Visual understanding (diagrams, graphs, geometric figures)
- German language comprehension
- Multi-choice question answering (5 options: A-E)
- Age-appropriate difficulty levels (grades 3-13)

All tasks include ground truth answers for automated evaluation.

### Benchmark Setup (DFKI Pegasus Cluster)

Das Benchmark nutzt den NVIDIA PyTorch Container (`nvcr.io_nvidia_pytorch_23.12-py3.sqsh`) mit PyTorch 2.1.0 und CUDA 12.3. Zusätzliche Pakete werden über ein Task-Prolog-Skript installiert.

**1. HuggingFace Token konfigurieren:**
```bash
echo 'HF_TOKEN=hf_your_token_here' > .env
```

**2. Job starten (Beispiel Qwen2.5-VL-3B):**
```bash
sbatch scripts/run_qwen2_5_vl_3b.sh
```

Das Skript:
- Nutzt `--task-prolog` um Dependencies einmalig zu installieren (`scripts/install.sh`)
- Erbt PyTorch/CUDA aus dem Container (keine manuelle Installation nötig)
- Speichert Logs in `evaluation_results/logs/`

**3. Fortschritt überwachen:**
```bash
squeue -u $USER
tail -f vlm_qwen2_5_vl_3b_*.out
```

### Skript-Struktur

```
scripts/
├── install.sh                 # Task-Prolog: Installiert Dependencies
├── run_qwen2_5_vl_3b.sh       # SLURM-Job für Qwen2.5-VL-3B
└── ...                        # Weitere Modell-Skripte

src/eval/models/
├── run_qwen2_5_vl_3b.py       # Python Benchmark-Skript
└── ...                        # Weitere Modelle
```

### Container-Details

| Eigenschaft | Wert |
|-------------|------|
| Image | `/enroot/nvcr.io_nvidia_pytorch_23.12-py3.sqsh` |
| PyTorch | 2.1.0 |
| CUDA | 12.3 |
| Python | 3.10 |

Zusätzlich installierte Pakete (via `install.sh`):
- `transformers>=4.44.0`
- `accelerate>=0.33.0`
- `qwen-vl-utils>=0.0.8`
- `bitsandbytes>=0.43.0`
- `pydantic`, `python-dotenv`, `pandas`, `pillow`

### Evaluated Models

| Model | Parameters | Quantization |
|-------|------------|--------------|
| Qwen2.5-VL-3B | 3B | FP16/BF16 |
| Qwen2.5-VL-7B | 7B | FP16/BF16 |
| Qwen2.5-VL-32B | 32B | FP16/BF16 |
| Qwen2.5-VL-72B | 72B | 4-Bit |
| InternVL3-8B | 8B | FP16/BF16 |
| InternVL3-14B | 14B | FP16/BF16 |
| InternVL3-38B | 38B | FP16/BF16 |
| InternVL3-78B | 78B | 4-Bit |

### Evaluation Output

```
evaluation_results/
├── {MODEL}_results.jsonl     # Detaillierte Ergebnisse pro Task
├── {MODEL}_summary.xlsx      # Excel-Zusammenfassung
├── {MODEL}.log               # Python-Logs
└── logs/                     # SLURM stdout/stderr
```

## 🔍 Technical Notes

### PDF Extraction Challenges (1998-2011)
- **Encoding issues:** PDFs have non-standard character encoding
- **Solution:** OCR-based extraction using Tesseract with German language support
- **Marker detection:** Different terminology between years:
  - 1998: "3-Punkte-Fragen" or "6-Punkte-Fragen" (Grade 3-4)
  - 2000-2011: "3-Punkte-Aufgaben"
- **Known issues:** Some tasks missing due to OCR detection failures

### Dataset Completeness
See [FEHLENDE_LÖSUNGEN.md](FEHLENDE_LÖSUNGEN.md) and [FEHLENDE_AUFGABEN_1998_2011.md](FEHLENDE_AUFGABEN_1998_2011.md) for documentation of missing data.
