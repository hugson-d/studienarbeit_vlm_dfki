# Zero-Shot VLM Evaluation on Kangaroo Math Problems

This repository is the paper artifact for a zero-shot evaluation of vision-language models on German Kangaroo Math multiple-choice problems. It is organized to make the zero-shot workflow easy to inspect, rerun, and cite.

The public-facing path is centered on four pieces:

- `data/final/`: final dataset manifest, images, and metadata used for evaluation
- `results/zero_shot/`: zero-shot raw predictions, summaries, and paper figures
- `scripts/inference/zero_shot/`: cluster launchers for the active zero-shot runs
- `notebooks/zero_shot_analysis.ipynb`: plotting and exploratory analysis notebook

Legacy material that is not part of the main paper narrative is preserved under archive paths instead of being mixed into the main workflow.

## Repository Layout

```text
.
├── data/
│   ├── final/
│   │   ├── dataset.json
│   │   ├── images/
│   │   ├── metadata.jsonl
│   │   ├── solutions_1998_2011.json
│   │   └── solutions_2012_2025.json
│   └── archive/
├── docs/
│   └── cluster.md
├── notebooks/
│   └── zero_shot_analysis.ipynb
├── results/
│   ├── zero_shot/
│   │   ├── raw/
│   │   ├── summary/
│   │   └── figures/
│   └── archive/
├── scripts/
│   ├── inference/
│   │   └── zero_shot/
│   └── archive/
└── src/
    ├── analysis/
    ├── eval/
    │   └── zero_shot/
    └── archive/
```

## Dataset

The evaluation dataset is stored in:

- `data/final/dataset.json`: task manifest
- `data/final/images/`: image files referenced by the manifest
- `data/final/metadata.jsonl`: additional metadata export

Each dataset entry contains the task year, class level, task ID, answer, category, text-only flag, extracted question text, and a relative image path rooted at `data/`.

## Zero-Shot Results

The main paper results are stored in:

- `results/zero_shot/raw/`: per-model zero-shot JSONL outputs
- `results/zero_shot/summary/`: aggregated CSV summaries
- `results/zero_shot/figures/`: exported paper figures

Archived non-zero-shot and side-experiment outputs are stored separately in `results/archive/`.

### Accuracy by Mathematical Category

Zero-shot accuracy by mathematical category. Accuracy is reported in percent and rounded to one decimal place. The best value per column is in **bold**.

| Model | All | Alg. | Arith. | Geom. | Stoch. | Unknown |
| --- | --- | --- | --- | --- | --- | --- |
| ***Qwen Family*** | | | | | | |
| Qwen3-VL-235B-A22B | 44.9 | 49.1 | 48.9 | 40.8 | 42.5 | 38.3 |
| Qwen3-VL-32B | 36.9 | 40.8 | 38.4 | 34.7 | 36.3 | 26.6 |
| Qwen3-VL-30B | 30.5 | 30.2 | 30.6 | 30.5 | 30.0 | 34.0 |
| Qwen3-VL-8B | 28.0 | 31.2 | 26.8 | 27.4 | 29.0 | 22.3 |
| Qwen3-VL-4B | 26.6 | 31.7 | 25.4 | 25.2 | 27.7 | 20.2 |
| Qwen2.5-VL-72B | 40.4 | 41.0 | 45.3 | 35.3 | 40.8 | 39.4 |
| Qwen2.5-VL-32B | 37.9 | 39.6 | 40.0 | 34.2 | 38.8 | 41.5 |
| Qwen2.5-VL-7B | 28.6 | 29.0 | 28.7 | 27.0 | 31.5 | 25.5 |
| Qwen2.5-VL-3B | 25.2 | 25.6 | 25.5 | 24.6 | 25.6 | 25.5 |
| ***Intern Family*** | | | | | | |
| InternVL3-38B | 31.3 | 31.7 | 31.2 | 29.1 | 35.1 | 31.9 |
| InternVL3-14B | 27.5 | 26.5 | 28.5 | 26.5 | 27.3 | 34.0 |
| InternVL3-8B | 22.7 | 21.6 | 22.7 | 22.9 | 22.6 | 27.7 |
| ***Ovis Family*** | | | | | | |
| Ovis2.5-9B | 31.1 | 33.1 | 30.8 | 29.4 | 32.0 | 35.1 |
| Ovis2.5-2B | 25.4 | 24.6 | 23.4 | 28.1 | 24.4 | 27.7 |
| ***Other Models (US/EU)*** | | | | | | |
| Idefics3-8B-Llama3 | 19.9 | 21.1 | 19.5 | 20.8 | 18.6 | 14.9 |
| Gemma-4-31B | 42.5 | 43.7 | 45.3 | 37.2 | 45.9 | 49.0 |
| Gemma-3-27B | 22.1 | 21.4 | 21.7 | 22.7 | 22.9 | 20.2 |
| Gemma-3-12B | 23.0 | 24.4 | 21.5 | 23.5 | 22.4 | 29.8 |
| Gemma-3-4B | 21.2 | 21.2 | 19.1 | 21.2 | 25.5 | 18.1 |
| OpenAI GPT-5.4 | **80.0** | **89.5** | **92.7** | **61.6** | **84.8** | **67.0** |
| Mistral Large 3 | 62.6 | 75.5 | 77.7 | 43.7 | 61.9 | 44.7 |
| Mistral Medium 3.1 | 47.6 | 55.5 | 60.0 | 35.6 | 42.5 | 33.0 |
| Mistral Small 3.1 | 49.0 | 55.0 | 65.9 | 34.8 | 41.3 | 34.8 |

### Accuracy by Difficulty Level

Zero-shot accuracy by difficulty level (A/B/C). Accuracy is reported in percent and rounded to one decimal place. The best value per column is in **bold**.

| Model | All | A | B | C |
| --- | --- | --- | --- | --- |
| ***Qwen Family*** | | | | |
| Qwen3-VL-235B-A22B | 44.9 | 50.3 | 43.8 | 40.7 |
| Qwen3-VL-32B | 36.9 | 42.4 | 34.8 | 33.6 |
| Qwen3-VL-30B | 30.5 | 32.9 | 29.6 | 29.1 |
| Qwen3-VL-8B | 28.0 | 31.2 | 26.7 | 26.1 |
| Qwen3-VL-4B | 26.6 | 31.1 | 24.3 | 24.6 |
| Qwen2.5-VL-72B | 40.4 | 46.8 | 38.3 | 36.1 |
| Qwen2.5-VL-32B | 37.9 | 44.2 | 35.0 | 34.5 |
| Qwen2.5-VL-7B | 28.6 | 30.9 | 28.2 | 26.6 |
| Qwen2.5-VL-3B | 25.2 | 26.2 | 24.4 | 25.1 |
| ***Intern Family*** | | | | |
| InternVL3-38B | 31.3 | 33.3 | 31.3 | 29.2 |
| InternVL3-14B | 27.5 | 28.2 | 27.5 | 26.6 |
| InternVL3-8B | 22.7 | 22.9 | 23.6 | 21.6 |
| ***Ovis Family*** | | | | |
| Ovis2.5-9B | 31.1 | 34.0 | 30.1 | 29.1 |
| Ovis2.5-2B | 25.4 | 27.2 | 24.1 | 25.0 |
| ***Other Models (US/EU)*** | | | | |
| Idefics3-8B-Llama3 | 19.9 | 20.1 | 18.9 | 20.7 |
| Gemma-4-31B | 42.5 | 49.9 | 40.2 | 37.6 |
| Gemma-3-27B | 22.1 | 21.7 | 23.5 | 21.3 |
| Gemma-3-12B | 23.0 | 22.6 | 23.6 | 22.9 |
| Gemma-3-4B | 21.2 | 20.3 | 19.9 | 23.4 |
| OpenAI GPT-5.4 | **80.0** | **83.7** | **82.5** | **73.8** |
| Mistral Large 3 | 62.6 | 67.3 | 63.6 | 57.2 |
| Mistral Medium 3.1 | 47.6 | 56.0 | 49.8 | 37.2 |
| Mistral Small 3.1 | 49.0 | 58.8 | 50.4 | 37.6 |

## Reproducing the Workflow

Install the Python environment with your preferred toolchain. The repository includes `pyproject.toml` and `uv.lock`.

Generate or refresh the zero-shot summary CSV:

```bash
python src/analysis/analyze_accuracy.py
```

Recreate the main zero-shot figures:

```bash
python src/analysis/plot_modality_gap_dumbbell_acl.py
python src/analysis/plot_zero_shot_time_trend_acl.py
```

Open the analysis notebook:

```bash
jupyter notebook notebooks/zero_shot_analysis.ipynb
```

Submit a zero-shot cluster run, for example:

```bash
sbatch scripts/inference/zero_shot/run_qwen2_5_vl_3b_vllm.sh
```

## Notes

- The active zero-shot runners live in `src/eval/zero_shot/`.
- Older CoT, temperature-sweep, failure-analysis, and preprocessing code is preserved in `src/archive/` and `scripts/archive/`.
- Cluster-specific setup notes are documented in `docs/cluster.md`.
