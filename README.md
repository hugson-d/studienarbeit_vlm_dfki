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
