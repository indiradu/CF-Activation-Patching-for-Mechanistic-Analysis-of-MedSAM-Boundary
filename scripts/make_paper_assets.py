#!/usr/bin/env python3
"""Create paper-ready figures and tables from completed experiment CSVs."""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from pathlib import Path

matplotlib_cache = Path(tempfile.gettempdir()) / "miccai_workshop_matplotlib"
matplotlib_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


METRICS = ["dice", "boundary_f1", "hd95"]
METRIC_LABELS = {
    "dice": "Dice",
    "iou": "IoU",
    "boundary_f1": "Boundary F1",
    "hd95": "HD95",
    "area_error": "Area error",
}
PALETTE = {
    "Original": "#455A64",
    "CF-Seg": "#00796B",
    "CXR-SD": "#C56B2C",
    "Matched": "#00796B",
    "Reverse": "#78909C",
    "Non-matched": "#C62828",
    "Background/context": "#1565C0",
    "Boundary band": "#6A1B9A",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("paper/assets"))
    parser.add_argument("--table-dir", type=Path, default=Path("paper/tables"))
    return parser.parse_args()


def format_float(value: float) -> str:
    if abs(value) >= 10:
        return f"{value:.2f}"
    return f"{value:.3f}"


def write_markdown_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = df.to_markdown(index=False, floatfmt=".3f", disable_numparse=True)
    path.write_text(text + "\n", encoding="utf-8")


def write_latex_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = df.to_latex(index=False, float_format=lambda value: f"{value:.3f}")
    path.write_text(text, encoding="utf-8")


def clean_generator_means(path: Path) -> pd.DataFrame:
    means = pd.read_csv(path)
    labels = {"original": "Original", "cfseg": "CF-Seg", "cxr_sd": "CXR-SD"}
    means["condition"] = means["condition"].map(labels).fillna(means["condition"])
    order = ["Original", "CF-Seg", "CXR-SD"]
    means["condition"] = pd.Categorical(means["condition"], categories=order, ordered=True)
    return means.sort_values("condition")


def plot_generator_metrics(means: pd.DataFrame, output: Path) -> None:
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.15)
    fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.2))
    order = ["Original", "CF-Seg", "CXR-SD"]
    colors = [PALETTE[label] for label in order]

    for ax, metric in zip(axes, METRICS, strict=True):
        values = [float(means.loc[means["condition"] == label, metric].iloc[0]) for label in order]
        bars = ax.bar(order, values, color=colors, edgecolor="#222222", linewidth=0.7)
        ax.set_title(METRIC_LABELS[metric])
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=20)
        if metric == "hd95":
            ax.set_ylabel("Pixels, lower is better")
        else:
            ax.set_ylim(0, max(1.0, max(values) * 1.12))
            ax.set_ylabel("Score, higher is better")
        for bar, value in zip(bars, values, strict=True):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                format_float(value),
                ha="center",
                va="bottom",
                fontsize=8,
            )
        sns.despine(ax=ax)

    fig.suptitle("MedSAM segmentation on original and counterfactual chest X-rays", y=1.03, fontsize=12)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def load_cfseg_controls(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    label_map = {
        "MedSAM original": "Original",
        "matched_cf_to_original / whole": "Matched",
        "reverse_original_to_cf / whole": "Reverse",
        "nonmatched_cf_to_original / whole": "Non-matched",
        "matched_cf_to_original / background": "Background/context",
        "matched_cf_to_original / boundary": "Boundary band",
    }
    df = df[df["condition"].isin(label_map)].copy()
    df["plot_condition"] = df["condition"].map(label_map)
    return df


def load_cxrsd_controls(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    label_map = {
        "matched whole (CXR-SD into original)": "Matched",
        "reverse whole (original into CXR-SD)": "Reverse",
        "non-matched whole": "Non-matched",
        "matched background/context": "Background/context",
        "matched boundary band": "Boundary band",
    }
    df = df[df["condition"].isin(label_map)].copy()
    df["plot_condition"] = df["condition"].map(label_map)
    original = pd.read_csv("outputs/statistics/nih_effusion_50_generator_comparison/condition_means.csv")
    original = original[original["condition"] == "original"].copy()
    original["plot_condition"] = "Original"
    return pd.concat([original, df], ignore_index=True, sort=False)


def plot_pairing_controls(cfseg: pd.DataFrame, cxrsd: pd.DataFrame, output: Path) -> None:
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.0), sharex=False)
    datasets = [("CF-Seg counterfactual source", cfseg), ("CXR-SD counterfactual source", cxrsd)]
    order = ["Original", "Matched", "Reverse", "Non-matched"]

    for row_idx, (title, df) in enumerate(datasets):
        subset = df[df["plot_condition"].isin(order)].copy()
        for col_idx, metric in enumerate(METRICS):
            ax = axes[row_idx, col_idx]
            values = [float(subset.loc[subset["plot_condition"] == label, metric].iloc[0]) for label in order]
            bars = ax.bar(order, values, color=[PALETTE[label] for label in order], edgecolor="#222222", linewidth=0.7)
            if row_idx == 0:
                ax.set_title(METRIC_LABELS[metric])
            if col_idx == 0:
                ax.set_ylabel(title)
            if metric != "hd95":
                ax.set_ylim(0, max(1.0, max(values) * 1.15))
            else:
                ax.set_ylabel(f"{title}\nPixels, lower is better" if col_idx == 0 else "Pixels")
            ax.tick_params(axis="x", rotation=25)
            for bar, value in zip(bars, values, strict=True):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), format_float(value), ha="center", va="bottom", fontsize=7)
            sns.despine(ax=ax)

    fig.suptitle("Activation-patching controls: matched, reverse, and non-matched donors", y=1.01, fontsize=12)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_context_vs_boundary(cfseg: pd.DataFrame, cxrsd: pd.DataFrame, output: Path) -> None:
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
    rows = []
    for source, df in [("CF-Seg", cfseg), ("CXR-SD", cxrsd)]:
        for label in ["Background/context", "Boundary band"]:
            row = df[df["plot_condition"] == label].iloc[0].copy()
            row["source"] = source
            row["region"] = label
            rows.append(row)
    plot_df = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.2))
    for ax, metric in zip(axes, METRICS, strict=True):
        sns.barplot(
            data=plot_df,
            x="source",
            y=metric,
            hue="region",
            palette=[PALETTE["Background/context"], PALETTE["Boundary band"]],
            edgecolor="#222222",
            linewidth=0.7,
            ax=ax,
        )
        ax.set_title(METRIC_LABELS[metric])
        ax.set_xlabel("")
        if metric == "hd95":
            ax.set_ylabel("Pixels, lower is better")
        else:
            ax.set_ylim(0, max(1.0, float(plot_df[metric].max()) * 1.15))
            ax.set_ylabel("Score, higher is better")
        ax.legend_.remove()
        sns.despine(ax=ax)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.08))
    fig.suptitle("Context/background patches carry more corrective signal than boundary-band patches", y=1.0, fontsize=12)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_layer_heatmap(summary_path: Path, output: Path, title: str) -> None:
    df = pd.read_csv(summary_path)
    df = df[(df["control"] == "matched_cf_to_original") & (df["region"].isin(["whole", "background", "boundary"]))].copy()
    layer_order = ["block3", "block9", "neck"]
    region_order = ["whole", "background", "boundary"]
    df["layer"] = pd.Categorical(df["layer"], categories=layer_order, ordered=True)
    df["region"] = pd.Categorical(df["region"], categories=region_order, ordered=True)
    matrix = df.pivot_table(index="layer", columns="region", values="boundary_f1", observed=False).loc[layer_order, region_order]

    sns.set_theme(style="white", context="paper", font_scale=1.1)
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".3f",
        cmap="YlGnBu",
        linewidths=0.6,
        linecolor="white",
        cbar_kws={"label": "Boundary F1"},
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("Patched region")
    ax.set_ylabel("MedSAM layer")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_method_schematic(output: Path) -> None:
    sns.set_theme(style="white", context="paper", font_scale=1.0)
    fig, ax = plt.subplots(figsize=(10.5, 3.2))
    ax.axis("off")

    boxes = [
        ("Diseased CXR\n$x$", (0.04, 0.58), "#ECEFF1"),
        ("Pseudo-healthy\ncounterfactual $x_{cf}$", (0.04, 0.18), "#E0F2F1"),
        ("MedSAM\nforward pass", (0.28, 0.38), "#F5F5F5"),
        ("Patch activations\n$A_l(x) \\leftarrow A_l(x_{cf})$", (0.52, 0.38), "#FFF3E0"),
        ("Mask metrics\nDice / BF1 / HD95", (0.77, 0.38), "#E8EAF6"),
    ]
    for text, (x, y), color in boxes:
        rect = plt.Rectangle((x, y), 0.18, 0.22, facecolor=color, edgecolor="#222222", linewidth=1.2)
        ax.add_patch(rect)
        ax.text(x + 0.09, y + 0.11, text, ha="center", va="center", fontsize=10)

    arrows = [
        ((0.22, 0.69), (0.28, 0.52)),
        ((0.22, 0.29), (0.28, 0.45)),
        ((0.46, 0.49), (0.52, 0.49)),
        ((0.70, 0.49), (0.77, 0.49)),
    ]
    for start, end in arrows:
        ax.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "lw": 1.8, "color": "#222222"})

    control_text = "Controls: matched donor, reverse patching, non-matched donor, boundary band, background/context"
    ax.text(0.5, 0.08, control_text, ha="center", va="center", fontsize=10, color="#333333")
    ax.set_title("Counterfactual activation patching for MedSAM boundary failures", fontsize=13, pad=10)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_key_stats_table(path: Path, output_md: Path, output_tex: Path) -> None:
    stats = pd.read_csv(path)
    keep = stats[stats["metric"].isin(["dice", "boundary_f1", "hd95"])].copy()
    keep = keep[
        keep["comparison"].isin(
            [
                "cfseg vs original",
                "cxr_sd vs original",
                "cfseg vs cxr_sd",
            ]
        )
    ]
    keep["metric"] = keep["metric"].map(METRIC_LABELS)
    keep["mean diff"] = keep["mean_difference_candidate_minus_baseline"].map(lambda value: f"{value:.3f}")
    keep["95% CI"] = keep.apply(lambda row: f"[{row['bootstrap_ci_low']:.3f}, {row['bootstrap_ci_high']:.3f}]", axis=1)
    keep["p"] = keep["wilcoxon_p"].map(lambda value: f"{value:.2e}")
    keep["candidate_better_fraction"] = keep["candidate_better_fraction"].map(lambda value: f"{value:.2f}")
    table = keep[["comparison", "metric", "mean diff", "95% CI", "p", "candidate_better_fraction"]].rename(
        columns={"candidate_better_fraction": "better frac"}
    )
    write_markdown_table(table, output_md)
    write_latex_table(table, output_tex)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.table_dir.mkdir(parents=True, exist_ok=True)

    generator_means = clean_generator_means(Path("outputs/statistics/nih_effusion_50_generator_comparison/condition_means.csv"))
    cfseg_controls = load_cfseg_controls(Path("outputs/activation_probe/nih_effusion_50_controls_bw32/paper_key_control_table.csv"))
    cxrsd_controls = load_cxrsd_controls(Path("outputs/activation_probe/nih_effusion_50_prism_v2/paper_key_control_table.csv"))

    plot_generator_metrics(generator_means, args.output_dir / "figure_generator_metric_bars.png")
    plot_pairing_controls(cfseg_controls, cxrsd_controls, args.output_dir / "figure_activation_pairing_controls.png")
    plot_context_vs_boundary(cfseg_controls, cxrsd_controls, args.output_dir / "figure_context_vs_boundary_controls.png")
    plot_method_schematic(args.output_dir / "figure_method_schematic.png")
    plot_layer_heatmap(
        Path("outputs/activation_probe/nih_effusion_50_controls_bw32/summary_by_control_layer_region.csv"),
        args.output_dir / "figure_cfseg_layer_region_heatmap.png",
        "CF-Seg matched patching",
    )
    plot_layer_heatmap(
        Path("outputs/activation_probe/nih_effusion_50_prism_v2/summary_by_control_layer_region.csv"),
        args.output_dir / "figure_cxrsd_layer_region_heatmap.png",
        "CXR-SD matched patching",
    )

    qualitative_source = Path("outputs/figures/qualitative_generators_nih_effusion_50.png")
    if qualitative_source.exists():
        shutil.copy2(qualitative_source, args.output_dir / "figure_qualitative_generators_nih_effusion_50.png")

    generator_table = generator_means.rename(columns={column: METRIC_LABELS.get(column, column) for column in generator_means.columns})
    write_markdown_table(generator_table, args.table_dir / "table_generator_metrics.md")
    write_latex_table(generator_table, args.table_dir / "table_generator_metrics.tex")

    cfseg_table = cfseg_controls[["plot_condition", "dice", "boundary_f1", "hd95", "area_error"]].rename(
        columns={"plot_condition": "condition", **METRIC_LABELS}
    )
    cxrsd_table = cxrsd_controls[["plot_condition", "dice", "boundary_f1", "hd95", "area_error"]].rename(
        columns={"plot_condition": "condition", **METRIC_LABELS}
    )
    write_markdown_table(cfseg_table, args.table_dir / "table_cfseg_activation_controls.md")
    write_latex_table(cfseg_table, args.table_dir / "table_cfseg_activation_controls.tex")
    write_markdown_table(cxrsd_table, args.table_dir / "table_cxrsd_activation_controls.md")
    write_latex_table(cxrsd_table, args.table_dir / "table_cxrsd_activation_controls.tex")

    make_key_stats_table(
        Path("outputs/statistics/nih_effusion_50_generator_comparison/paired_bootstrap_wilcoxon.csv"),
        args.table_dir / "table_generator_bootstrap_stats.md",
        args.table_dir / "table_generator_bootstrap_stats.tex",
    )

    print(f"Wrote figures to {args.output_dir}")
    print(f"Wrote tables to {args.table_dir}")


if __name__ == "__main__":
    main()
