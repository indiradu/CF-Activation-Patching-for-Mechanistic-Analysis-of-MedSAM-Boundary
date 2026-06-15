#!/usr/bin/env python3
"""Summarize supervisor-requested robustness experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


METRICS = ["dice", "iou", "boundary_f1", "hd95", "area_error"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/supervisor_robustness"))
    parser.add_argument(
        "--activation-suffix",
        default="",
        help="Optional suffix for activation output directories, e.g. _neck.",
    )
    return parser.parse_args()


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def metric_means(path: Path) -> dict[str, float]:
    df = read_csv(path)
    return {metric: float(df[metric].mean()) for metric in METRICS}


def patch_means(path: Path, *, control: str = "matched_cf_to_original") -> pd.DataFrame:
    df = read_csv(path)
    df = df[df["control"] == control].copy()
    return (
        df.groupby("region", as_index=False)[METRICS]
        .mean()
        .sort_values(["boundary_f1", "dice"], ascending=[False, False])
    )


def patch_condition(path: Path, *, control: str, region: str) -> dict[str, float]:
    df = read_csv(path)
    subset = df[(df["control"] == control) & (df["region"] == region)].copy()
    if subset.empty:
        return {metric: float("nan") for metric in METRICS}
    return {metric: float(subset[metric].mean()) for metric in METRICS}


def add_prefixed(row: dict[str, object], prefix: str, values: dict[str, float]) -> None:
    for metric, value in values.items():
        row[f"{prefix}_{metric}"] = value


def prompt_jitter_summary(project_root: Path, activation_suffix: str = "") -> pd.DataFrame:
    rows = []
    for label, pct in [("05", 0.05), ("10", 0.10), ("15", 0.15)]:
        original_metrics = project_root / f"outputs/medsam/nih_effusion_50/prompt_jitter_{label}/original/metrics_vs_chexmask.csv"
        cfseg_metrics = project_root / f"outputs/medsam/nih_effusion_50/prompt_jitter_{label}/cfseg/metrics_vs_chexmask.csv"
        patching = project_root / f"outputs/activation_probe/nih_effusion_50_prompt_jitter_{label}{activation_suffix}/patching_metrics.csv"
        row: dict[str, object] = {"jitter_pct": pct}
        add_prefixed(row, "original", metric_means(original_metrics))
        add_prefixed(row, "cfseg", metric_means(cfseg_metrics))
        add_prefixed(row, "matched_whole", patch_condition(patching, control="matched_cf_to_original", region="whole"))
        add_prefixed(row, "nonmatched_whole", patch_condition(patching, control="nonmatched_cf_to_original", region="whole"))
        add_prefixed(row, "background", patch_condition(patching, control="matched_cf_to_original", region="background"))
        add_prefixed(row, "boundary", patch_condition(patching, control="matched_cf_to_original", region="boundary"))
        row["matched_minus_nonmatched_boundary_f1"] = row["matched_whole_boundary_f1"] - row["nonmatched_whole_boundary_f1"]
        row["background_minus_boundary_boundary_f1"] = row["background_boundary_f1"] - row["boundary_boundary_f1"]
        rows.append(row)
    return pd.DataFrame(rows)


def boundary_width_summary(project_root: Path, activation_suffix: str = "") -> pd.DataFrame:
    specs = [
        ("CF-Seg", 16, project_root / f"outputs/activation_probe/nih_effusion_50_controls_bw16{activation_suffix}/patching_metrics.csv"),
        ("CF-Seg", 32, project_root / f"outputs/activation_probe/nih_effusion_50_controls_bw32{activation_suffix}/patching_metrics.csv"),
        ("CF-Seg", 64, project_root / f"outputs/activation_probe/nih_effusion_50_controls_bw64{activation_suffix}/patching_metrics.csv"),
        ("CXR-SD", 16, project_root / f"outputs/activation_probe/nih_effusion_50_prism_v2_random_bw16{activation_suffix}/patching_metrics.csv"),
        ("CXR-SD", 32, project_root / f"outputs/activation_probe/nih_effusion_50_prism_v2_random_bw32{activation_suffix}/patching_metrics.csv"),
        ("CXR-SD", 64, project_root / f"outputs/activation_probe/nih_effusion_50_prism_v2_random_bw64{activation_suffix}/patching_metrics.csv"),
    ]
    rows = []
    for source, width, path in specs:
        means = patch_means(path)
        for _, row in means.iterrows():
            rows.append(
                {
                    "source": source,
                    "boundary_width_px": width,
                    "region": row["region"],
                    **{metric: float(row[metric]) for metric in METRICS},
                }
            )
    return pd.DataFrame(rows)


def compact_width_table(width_df: pd.DataFrame) -> pd.DataFrame:
    keep = width_df[width_df["region"].isin(["whole", "background", "boundary", "random_like_boundary", "random_like_background"])]
    return keep[
        [
            "source",
            "boundary_width_px",
            "region",
            "dice",
            "boundary_f1",
            "hd95",
            "area_error",
        ]
    ].sort_values(["source", "boundary_width_px", "region"])


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    output_dir = (project_root / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    prompt = prompt_jitter_summary(project_root, activation_suffix=args.activation_suffix)
    width = boundary_width_summary(project_root, activation_suffix=args.activation_suffix)
    compact_width = compact_width_table(width)

    prompt.to_csv(output_dir / "prompt_jitter_summary.csv", index=False)
    width.to_csv(output_dir / "boundary_width_summary.csv", index=False)
    compact_width.to_csv(output_dir / "boundary_width_compact_table.csv", index=False)
    print(f"Wrote robustness summaries to {output_dir}")
    print("\nPrompt jitter:")
    print(prompt[["jitter_pct", "original_boundary_f1", "cfseg_boundary_f1", "matched_whole_boundary_f1", "nonmatched_whole_boundary_f1", "background_boundary_f1", "boundary_boundary_f1"]].to_string(index=False))
    print("\nBoundary width compact:")
    print(compact_width.to_string(index=False))


if __name__ == "__main__":
    main()
