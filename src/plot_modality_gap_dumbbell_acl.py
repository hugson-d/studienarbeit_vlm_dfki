#!/usr/bin/env python3
"""Create an ACL-style dumbbell chart for zero-shot modality gaps."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EXCLUDE_NAME_PARTS = (
    "CoT-Voting",
    "_n1",
    "_n5",
    "TempSweep",
    "results_2025_non_cot_voting",
    "null_prediction",
    "with_prediction",
)


def is_zero_shot_file(path: Path) -> bool:
    name = path.name
    if not name.endswith(".jsonl"):
        return False
    return not any(part in name for part in EXCLUDE_NAME_PARTS)


def clean_model_name(model: str) -> str:
    lower = model.lower()
    if lower == "gpt-5.4":
        return "GPT-5.4"
    if lower == "mistral-large-2512":
        return "Mistral Large"
    if lower == "mistral-medium-2508":
        return "Mistral Medium"
    if lower == "mistral-small-2506":
        return "Mistral Mini"

    cleaned = model
    for suffix in ("-vLLM", "-Instruct", "-AWQ"):
        cleaned = cleaned.replace(suffix, "")
    return cleaned


def load_zero_shot_df(results_dir: Path) -> pd.DataFrame:
    files = sorted([p for p in results_dir.glob("*.jsonl") if is_zero_shot_file(p)])
    if not files:
        raise ValueError(f"No zero-shot JSONL files found in: {results_dir}")

    frames: list[pd.DataFrame] = []
    for fpath in files:
        df = pd.read_json(fpath, lines=True)
        needed = {"model", "is_text_only", "is_correct"}
        missing = needed - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns {missing} in {fpath.name}")
        frames.append(df)

    data = pd.concat(frames, ignore_index=True)
    if "math_category" in data.columns:
        data = data[data["math_category"].fillna("unknown").ne("unknown")].copy()

    data["is_text_only"] = data["is_text_only"].astype(bool)
    data["is_correct"] = data["is_correct"].astype(bool)
    data["model_clean"] = data["model"].map(clean_model_name)
    return data


def compute_plot_data(df: pd.DataFrame) -> pd.DataFrame:
    agg = (
        df.groupby(["model_clean", "is_text_only"], dropna=False)["is_correct"]
        .mean()
        .mul(100.0)
        .reset_index()
    )
    agg["modality"] = np.where(agg["is_text_only"], "Text-only", "Visual")
    wide = (
        agg.pivot(index="model_clean", columns="modality", values="is_correct")
        .reindex(columns=["Text-only", "Visual"])
        .dropna(subset=["Text-only", "Visual"])
        .copy()
    )
    wide["Gap_pp"] = wide["Text-only"] - wide["Visual"]
    wide = wide.sort_values("Gap_pp", ascending=False)
    return wide


def plot_modality_gap_dumbbell(
    plot_data: pd.DataFrame,
    title: str,
    savepath: Path,
    dpi: int,
) -> None:
    df = plot_data.copy()
    models = df.index.to_list()
    y = np.arange(len(models))

    text = df["Text-only"].to_numpy(dtype=float)
    vis = df["Visual"].to_numpy(dtype=float)
    gap = df["Gap_pp"].to_numpy(dtype=float)

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
        }
    )

    fig, ax = plt.subplots(figsize=(7.2, max(4.2, 0.24 * len(models))))

    for i in range(len(models)):
        ax.plot([vis[i], text[i]], [y[i], y[i]], color="0.35", linewidth=1.2, alpha=0.75, zorder=1)

    ax.scatter(vis, y, s=36, marker="o", linewidths=0.8, label="Visual", zorder=3)
    ax.scatter(text, y, s=36, marker="s", linewidths=0.8, label="Text-only", zorder=3)

    x_max = float(max(df["Text-only"].max(), df["Visual"].max()))
    pad = 3.0
    for i in range(len(models)):
        ax.text(x_max + pad, y[i], f"Δ={gap[i]:+,.1f} pp", va="center", ha="left", fontsize=8)

    ax.set_yticks(y)
    ax.set_yticklabels(models)
    ax.invert_yaxis()

    ax.set_xlabel("Accuracy (%)")
    ax.set_title(title, fontsize=10, pad=8)
    ax.set_xlim(0, min(100, x_max + pad + 16))
    ax.grid(axis="x", linestyle="--", linewidth=0.8, alpha=0.35)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1.14, 1.0),
        borderaxespad=0.0,
    )

    fig.tight_layout()
    fig.subplots_adjust(right=0.66)
    savepath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(savepath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot zero-shot modality gap dumbbell chart for all available models."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results_vllm"),
        help="Directory with JSONL result files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("img_results/modality_gap_dumbbell_acl_all_zeroshot.png"),
        help="Output PNG path.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=400,
        help="PNG export DPI.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = load_zero_shot_df(args.results_dir)
    plot_data = compute_plot_data(df)

    title = "Zero-shot modality gap by model (Text-only vs Visual)"
    plot_modality_gap_dumbbell(
        plot_data=plot_data,
        title=title,
        savepath=args.output,
        dpi=args.dpi,
    )
    print(f"Saved figure to: {args.output}")
    print(f"Models plotted: {len(plot_data)}")


if __name__ == "__main__":
    main()
