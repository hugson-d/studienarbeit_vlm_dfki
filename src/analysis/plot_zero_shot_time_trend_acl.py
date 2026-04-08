#!/usr/bin/env python3
"""Create an ACL-style zero-shot accuracy-over-time figure (PNG).

The script loads non-CoT result JSONL files for selected models, computes pooled
accuracy by model x year with Wilson 95% confidence intervals, and renders a
2x1 panel plot:
(a) Text-only items
(b) Visual items
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DEFAULT_MODELS = [
    "Qwen3-VL-30B-Instruct",
    "Qwen3-VL-8B-Instruct",
    "Qwen3-VL-4B-Instruct",
    "mistral-large-2512",
    "gpt-5.4",
]

# Explicit mapping to non-CoT zero-shot files only.
DEFAULT_FILE_MAP = {
    "Qwen3-VL-30B-Instruct": "Qwen3-VL-30B-Instruct_results.jsonl",
    "Qwen3-VL-8B-Instruct": "Qwen3-VL-8B-Instruct_results.jsonl",
    "Qwen3-VL-4B-Instruct": "Qwen3-VL-4B-Instruct_results.jsonl",
    "mistral-large-2512": "mistral-large-2512_structured_api_results.jsonl",
    "mistral-medium-2508": "mistral-medium-2508_structured_api_results.jsonl",
    "mistral-small-2506": "mistral-small-2506_structured_api_results.jsonl",
    "gpt-5.4": "gpt-5.4_results.jsonl",
}

DISPLAY_NAME = {
    "Qwen3-VL-30B-Instruct": "Qwen3-VL-30B",
    "Qwen3-VL-8B-Instruct": "Qwen3-VL-8B",
    "Qwen3-VL-4B-Instruct": "Qwen3-VL-4B",
    "mistral-large-2512": "Mistral Large",
    "mistral-medium-2508": "Mistral Medium",
    "mistral-small-2506": "Mistral Mini",
    "gpt-5.4": "GPT-5.4",
}

LINESTYLES = [
    "solid",
    "dashed",
    "dashdot",
    (0, (1, 1)),
    (0, (3, 1, 1, 1)),
    (0, (5, 2)),
    (0, (2, 2, 2, 4)),
]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (np.nan, np.nan)
    phat = k / n
    denom = 1.0 + (z**2) / n
    center = (phat + (z**2) / (2.0 * n)) / denom
    half = (z * np.sqrt((phat * (1.0 - phat)) / n + (z**2) / (4.0 * n**2))) / denom
    return (center - half, center + half)


def load_zero_shot_df(results_dir: Path, model_order: Iterable[str]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    missing_files: list[str] = []

    for model in model_order:
        fname = DEFAULT_FILE_MAP[model]
        fpath = results_dir / fname
        if not fpath.exists():
            missing_files.append(str(fpath))
            continue
        frames.append(pd.read_json(fpath, lines=True))

    if missing_files:
        missing = "\n".join(missing_files)
        raise FileNotFoundError(f"Missing required zero-shot files:\n{missing}")

    if not frames:
        raise ValueError("No result data loaded.")

    df = pd.concat(frames, ignore_index=True)
    required = {"model", "year", "is_text_only", "is_correct"}
    missing_cols = sorted(required - set(df.columns))
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    df = df[df["model"].isin(list(model_order))].copy()
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df[df["year"].notna()].copy()
    df["year"] = df["year"].astype(int)

    if "math_category" in df.columns:
        df = df[df["math_category"].fillna("unknown").ne("unknown")].copy()

    df["is_text_only"] = df["is_text_only"].astype(bool)
    df["is_correct"] = df["is_correct"].astype(bool)
    return df


def agg_acc_over_time(df: pd.DataFrame, models: list[str], text_only: bool) -> pd.DataFrame:
    d = df[(df["model"].isin(models)) & (df["is_text_only"] == bool(text_only))].copy()
    g = d.groupby(["model", "year"]) ["is_correct"]
    out = g.agg(k="sum", n="size").reset_index()
    out["acc"] = (out["k"] / out["n"]) * 100.0

    ci = out.apply(lambda r: wilson_ci(int(r["k"]), int(r["n"])), axis=1)
    out["lo"] = [c[0] * 100.0 for c in ci]
    out["hi"] = [c[1] * 100.0 for c in ci]

    out["year"] = out["year"].astype(int)
    return out.sort_values(["model", "year"]).reset_index(drop=True)


def plot_time_trend(
    ax: plt.Axes,
    agg_df: pd.DataFrame,
    models: list[str],
    years_all: list[int],
    linestyles: list,
    markers: list[str],
    display_name: dict[str, str],
    panel_label: str,
    y_min: float,
    y_max: float,
) -> tuple[list, list]:
    ax.set_ylabel("Accuracy (%)", fontsize=10)
    ax.grid(axis="y", linestyle="--", linewidth=0.8, alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    handles, labels = [], []

    for i, m in enumerate(models):
        dm = agg_df[agg_df["model"] == m].copy()
        if dm.empty:
            continue

        dm = dm.set_index("year").reindex(years_all)

        x = np.array(years_all, dtype=int)
        y = dm["acc"].to_numpy(dtype=float)

        line = ax.plot(
            x,
            y,
            linewidth=1.8,
            linestyle=linestyles[i % len(linestyles)],
            marker=markers[i % len(markers)],
            markersize=3.8,
            label=display_name.get(m, m),
        )[0]

        lo = dm["lo"].to_numpy(dtype=float)
        hi = dm["hi"].to_numpy(dtype=float)
        ax.fill_between(x, lo, hi, alpha=0.10, color=line.get_color(), linewidth=0)

        handles.append(line)
        labels.append(display_name.get(m, m))

    ax.set_xticks(years_all)
    ax.set_xlim(min(years_all) - 0.5, max(years_all) + 0.5)
    ax.tick_params(axis="x", labelrotation=90, labelsize=8)
    ax.tick_params(axis="y", labelsize=8)
    ax.set_ylim(y_min, y_max)

    ax.text(
        0.0,
        1.02,
        panel_label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
        fontweight="bold",
    )

    return handles, labels


def build_figure(df: pd.DataFrame, models: list[str], output_path: Path, dpi: int, show_title: bool) -> None:
    agg_text = agg_acc_over_time(df, models=models, text_only=True)
    agg_vis = agg_acc_over_time(df, models=models, text_only=False)

    years_all = sorted(
        set(agg_text["year"].dropna().astype(int).unique())
        | set(agg_vis["year"].dropna().astype(int).unique())
    )
    if not years_all:
        raise ValueError("No valid yearly data found after filtering.")

    global_max = np.nanmax([agg_text["hi"].max(), agg_vis["hi"].max()])
    y_max = float(np.ceil(global_max / 5.0) * 5.0)
    y_min = 0.0

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "axes.titlesize": 10,
            "axes.labelsize": 10,
            "legend.fontsize": 8,
        }
    )

    # Full-width ACL-style figure proportions.
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 4.8), sharex=True, sharey=True)

    h1, lab1 = plot_time_trend(
        axes[0],
        agg_text,
        models,
        years_all,
        linestyles=LINESTYLES,
        markers=MARKERS,
        display_name=DISPLAY_NAME,
        panel_label="(a) Text-only",
        y_min=y_min,
        y_max=y_max,
    )
    plot_time_trend(
        axes[1],
        agg_vis,
        models,
        years_all,
        linestyles=LINESTYLES,
        markers=MARKERS,
        display_name=DISPLAY_NAME,
        panel_label="(b) Visual",
        y_min=y_min,
        y_max=y_max,
    )

    axes[1].set_xlabel("Task Year", fontsize=10)

    fig.legend(
        h1,
        lab1,
        frameon=False,
        ncol=1,
        loc="center left",
        bbox_to_anchor=(0.83, 0.5),
        borderaxespad=0.0,
    )

    if show_title:
        fig.suptitle(
            "Zero-shot performance by task year (pooled accuracy with 95% Wilson CI)",
            fontsize=10,
            y=0.995,
        )
        fig.tight_layout(rect=[0.03, 0.04, 0.80, 0.95])
    else:
        fig.tight_layout(rect=[0.03, 0.04, 0.80, 0.98])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create an ACL-style 2x1 plot of zero-shot accuracy over task year "
            "for selected models."
        )
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/zero_shot/raw"),
        help="Directory containing result JSONL files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/zero_shot/figures/zero_shot_time_trend_acl.png"),
        help="Output PNG file path.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=400,
        help="Output DPI for PNG export.",
    )
    parser.add_argument(
        "--show-title",
        action="store_true",
        help="Include an in-figure title (disabled by default for ACL camera-ready style).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    models = DEFAULT_MODELS

    df = load_zero_shot_df(args.results_dir, model_order=models)

    # Quick sanity check for modality coverage.
    missing_tx = [m for m in models if df[(df["model"] == m) & (df["is_text_only"] == True)].empty]
    missing_vi = [m for m in models if df[(df["model"] == m) & (df["is_text_only"] == False)].empty]
    if missing_tx:
        print("Warning: no text-only rows for:", ", ".join(missing_tx))
    if missing_vi:
        print("Warning: no visual rows for:", ", ".join(missing_vi))

    build_figure(df=df, models=models, output_path=args.output, dpi=args.dpi, show_title=args.show_title)
    print(f"Saved figure to: {args.output}")


if __name__ == "__main__":
    main()
