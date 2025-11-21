# Känguru-Wettbewerb VLM Dataset

Structured dataset and tooling for evaluating Vision Language Models (VLMs) on German Känguru math competition tasks (2010-2025).

## 📊 Dataset Overview

- **2,060 tasks** ready for VLM evaluation (`dataset_final/`)
- **149 excluded tasks** (visual/quality issues) in `dataset_final_not_used/`
- **16 years** of competition data (2010-2025)
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

### OpenAI Vision API Tools

**Categorize math tasks** (adds `math_category` field):
```bash
uv run python src/categorize_math_tasks.py
```
Categories: Arithmetik, Algebra, Geometrie, Logik, Kombinatorik, etc.

**Analyze text-only tasks** (adds `is_text_only` field):
```bash
uv run python src/analyze_text_only.py
```
Determines if task can be solved without visual elements.

**Extract text from images** (adds `extracted_text` field):
```bash
uv run python src/extract_text.py
```
Extracts question text and answer options using GPT-4o-mini.

### Utility Scripts

**Map solutions from 1998-2011** (fills `answer` field):
```bash
uv run python src/util_mapping.py
```

**Convert numeric task IDs to ABC format** (for consistency):
```bash
uv run python src/util_convert_task_ids.py
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

## 🗃️ Repository Structure

```
├── dataset_final.json          # Main dataset (2060 tasks)
├── DATASET_STATS.md            # Auto-generated statistics
├── MAPPING_LOGIC.md            # Task ID conversion documentation
├── data/
│   ├── dataset_final/          # Task images (for evaluation)
│   ├── dataset_final_not_used/ # Excluded task images
│   ├── lösungen_1998_2011.json # Solutions 1998-2011
│   └── lösungen_2012_2025.json # Solutions 2012-2025
└── src/
    ├── analyze_dataset_distribution.py
    ├── categorize_math_tasks.py
    ├── analyze_text_only.py
    ├── extract_text.py
    ├── util_mapping.py
    └── util_convert_task_ids.py
```

## 🎯 Usage Rate

- **93.3%** of available tasks are usable for VLM evaluation
- Quality filtering removes tasks with visual artifacts, poor scans, or complex layouts
- See comparison in DATASET_STATS.md

## 📚 Documentation

- [DATASET_STATS.md](DATASET_STATS.md) - Detailed statistics and distributions
- [MAPPING_LOGIC.md](MAPPING_LOGIC.md) - Task ID format and conversion rules

## 🤖 VLM Evaluation

This dataset is designed for evaluating Vision Language Models on:
- Mathematical reasoning
- Visual understanding (diagrams, graphs)
- German language comprehension
- Multi-choice question answering

All tasks include ground truth answers for automated evaluation.
