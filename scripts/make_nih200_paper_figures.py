#!/usr/bin/env python3
"""Generate updated NIH-200 manuscript figures.

The figures are intentionally tied to the scaled 200-case CF-Seg and CXR-SD
experiments so the visual evidence matches the manuscript tables.
"""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

matplotlib_cache = Path(tempfile.gettempdir()) / "miccai_workshop_matplotlib"
matplotlib_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


METRICS = ("dice", "boundary_f1", "hd95")
METRIC_TITLES = {
    "dice": "Dice",
    "boundary_f1": "Boundary F1",
    "hd95": "HD95 reduction",
}
METRIC_UNITS = {
    "dice": "change, higher is better",
    "boundary_f1": "change, higher is better",
    "hd95": "pixels reduced, higher is better",
}
SOURCE_COLORS = {
    "CF-Seg": "#00796B",
    "CXR-SD": "#C56B2C",
}
BAR_COLORS = {
    "Original": "#455A64",
    "Matched whole": "#00796B",
    "Reverse whole": "#78909C",
    "Non-matched whole": "#C62828",
    "Background/context": "#1565C0",
    "Boundary band": "#6A1B9A",
    "Interior": "#8D6E63",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("paper_revision/assets"))
    parser.add_argument(
        "--cfseg-paired",
        type=Path,
        default=Path("outputs/statistics/nih_effusion_200_neck_controls_bw32/paired_case_metrics.csv"),
    )
    parser.add_argument(
        "--cxrsd-paired",
        type=Path,
        default=Path("outputs/statistics/nih_effusion_200_prism_v2_neck_controls_bw32/paired_case_metrics.csv"),
    )
    parser.add_argument(
        "--cfseg-means",
        type=Path,
        default=Path("outputs/statistics/nih_effusion_200_neck_controls_bw32/condition_means.csv"),
    )
    parser.add_argument(
        "--cxrsd-means",
        type=Path,
        default=Path("outputs/statistics/nih_effusion_200_prism_v2_neck_controls_bw32/condition_means.csv"),
    )
    return parser.parse_args()


def bootstrap_ci(values: np.ndarray, *, rng_seed: int = 17, n_boot: int = 5000) -> tuple[float, float]:
    rng = np.random.default_rng(rng_seed)
    values = np.asarray(values, dtype=float)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[idx].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def improvement_delta(df: pd.DataFrame, candidate_prefix: str, metric: str) -> np.ndarray:
    original = df[f"original__{metric}"].to_numpy(dtype=float)
    candidate = df[f"{candidate_prefix}__{metric}"].to_numpy(dtype=float)
    if metric == "hd95":
        return original - candidate
    return candidate - original


def plot_per_case_deltas(cfseg_path: Path, cxrsd_path: Path, output: Path) -> None:
    sources = [
        ("CF-Seg", pd.read_csv(cfseg_path), "cfseg_counterfactual"),
        ("CXR-SD", pd.read_csv(cxrsd_path), "cxr_sd"),
    ]

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 8,
        }
    )
    fig, axes = plt.subplots(2, 3, figsize=(10.2, 5.4), constrained_layout=True)

    for row_idx, (source, df, prefix) in enumerate(sources):
        color = SOURCE_COLORS[source]
        for col_idx, metric in enumerate(METRICS):
            ax = axes[row_idx, col_idx]
            deltas = improvement_delta(df, prefix, metric)
            sorted_deltas = np.sort(deltas)
            x = np.arange(1, len(sorted_deltas) + 1)
            mean = float(deltas.mean())
            ci_low, ci_high = bootstrap_ci(deltas, rng_seed=100 + row_idx * 10 + col_idx)

            ax.scatter(x, sorted_deltas, s=9, color=color, alpha=0.72, linewidths=0)
            ax.axhline(0, color="#333333", linewidth=0.8, linestyle="--")
            ax.axhline(mean, color="#111111", linewidth=1.1)
            ax.fill_between([1, len(sorted_deltas)], ci_low, ci_high, color="#111111", alpha=0.08)
            ax.text(
                0.03,
                0.94,
                f"mean {mean:+.3f}" if metric != "hd95" else f"mean {mean:+.1f}px",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=8,
                bbox={"boxstyle": "round,pad=0.22", "facecolor": "white", "edgecolor": "#BBBBBB", "alpha": 0.9},
            )

            if row_idx == 0:
                ax.set_title(METRIC_TITLES[metric])
            if col_idx == 0:
                ax.set_ylabel(f"{source}\n{METRIC_UNITS[metric]}")
            else:
                ax.set_ylabel(METRIC_UNITS[metric])
            if row_idx == 1:
                ax.set_xlabel("Cases sorted by paired improvement")
            ax.grid(axis="y", color="#DDDDDD", linewidth=0.6)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

    fig.suptitle("Per-case MedSAM improvement from patient-matched counterfactual images", fontsize=12)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def condition_lookup(path: Path) -> dict[str, dict[str, float]]:
    rows = pd.read_csv(path).set_index("condition")
    return rows.to_dict(orient="index")


def bar_group(
    ax: plt.Axes,
    cfseg: dict[str, dict[str, float]],
    cxrsd: dict[str, dict[str, float]],
    conditions: list[tuple[str, str]],
    title: str,
) -> None:
    x = np.arange(len(conditions))
    width = 0.36
    cf_values = [cfseg[key]["boundary_f1"] for key, _ in conditions]
    cx_values = [cxrsd[key]["boundary_f1"] for key, _ in conditions]

    ax.bar(x - width / 2, cf_values, width, label="CF-Seg", color=SOURCE_COLORS["CF-Seg"], edgecolor="#222222", linewidth=0.6)
    ax.bar(x + width / 2, cx_values, width, label="CXR-SD", color=SOURCE_COLORS["CXR-SD"], edgecolor="#222222", linewidth=0.6)

    for xpos, value in zip(x - width / 2, cf_values, strict=True):
        ax.text(xpos, value + 0.006, f"{value:.3f}", ha="center", va="bottom", fontsize=7, rotation=90)
    for xpos, value in zip(x + width / 2, cx_values, strict=True):
        ax.text(xpos, value + 0.006, f"{value:.3f}", ha="center", va="bottom", fontsize=7, rotation=90)

    ax.set_xticks(x, [label for _, label in conditions], rotation=18, ha="right")
    ax.set_title(title)
    ax.set_ylabel("Boundary F1")
    ax.set_ylim(0, max(cf_values + cx_values) * 1.34)
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_neck_controls(cfseg_means: Path, cxrsd_means: Path, output: Path) -> None:
    cfseg = condition_lookup(cfseg_means)
    cxrsd = condition_lookup(cxrsd_means)

    whole_conditions = [
        ("original", "Original"),
        ("matched_whole", "Matched"),
        ("reverse_whole", "Reverse"),
        ("nonmatched_whole", "Non-matched"),
    ]
    region_conditions = [
        ("matched_background", "Background/context"),
        ("matched_boundary", "Boundary band"),
        ("matched_interior", "Interior"),
    ]

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.55), constrained_layout=True)
    bar_group(axes[0], cfseg, cxrsd, whole_conditions, "Whole-map causal controls")
    bar_group(axes[1], cfseg, cxrsd, region_conditions, "Regional neck controls")
    axes[1].legend(loc="upper right", frameon=False)
    fig.suptitle("Scaled 200-case activation patching at the MedSAM encoder neck", fontsize=12)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_per_case_deltas(
        args.cfseg_paired,
        args.cxrsd_paired,
        args.output_dir / "figure_per_case_deltas_nih200.png",
    )
    plot_neck_controls(
        args.cfseg_means,
        args.cxrsd_means,
        args.output_dir / "figure_activation_neck_controls_nih200.png",
    )
    print(f"Wrote updated NIH-200 figures to {args.output_dir}")


if __name__ == "__main__":
    main()
