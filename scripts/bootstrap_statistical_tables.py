#!/usr/bin/env python3
"""Create paired bootstrap and Wilcoxon tables for MedSAM experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


METRICS = ["dice", "iou", "boundary_f1", "hd95", "area_error"]
LOWER_IS_BETTER = {"hd95", "area_error"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-metrics", required=True, type=Path)
    parser.add_argument("--counterfactual-metrics", required=True, type=Path)
    parser.add_argument("--patching-metrics", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--case-column", default="case_id")
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def load_metric_frame(path: Path, label: str, case_column: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    keep = [case_column, *METRICS]
    missing = [column for column in keep if column not in df.columns]
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    df = df[keep].copy()
    df = df.rename(columns={metric: f"{label}__{metric}" for metric in METRICS})
    return df


def patch_condition(
    df: pd.DataFrame,
    *,
    control: str,
    region: str,
    label: str,
    case_column: str,
    layer: str | None = None,
) -> pd.DataFrame:
    subset = df[(df["control"] == control) & (df["region"] == region)].copy()
    if layer is not None:
        subset = subset[subset["layer"] == layer].copy()
    subset = subset.groupby(case_column, as_index=False)[METRICS].mean()
    subset = subset.rename(columns={metric: f"{label}__{metric}" for metric in METRICS})
    return subset[[case_column, *[f"{label}__{metric}" for metric in METRICS]]]


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator, samples: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    draws = rng.choice(values, size=(samples, len(values)), replace=True)
    means = draws.mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired_table(
    df: pd.DataFrame,
    *,
    case_column: str,
    comparisons: list[tuple[str, str]],
    bootstrap_samples: int,
    seed: int,
) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(seed)
    for baseline, candidate in comparisons:
        for metric in METRICS:
            base = df[f"{baseline}__{metric}"].to_numpy(dtype=float)
            cand = df[f"{candidate}__{metric}"].to_numpy(dtype=float)
            diff = cand - base
            ci_low, ci_high = bootstrap_ci(diff, rng, bootstrap_samples)
            try:
                test = wilcoxon(diff, zero_method="wilcox", alternative="two-sided")
                p_value = float(test.pvalue)
            except ValueError:
                p_value = float("nan")
            better = -diff if metric in LOWER_IS_BETTER else diff
            rows.append(
                {
                    "comparison": f"{candidate} vs {baseline}",
                    "baseline": baseline,
                    "candidate": candidate,
                    "metric": metric,
                    "n": int(len(diff)),
                    "baseline_mean": float(base.mean()),
                    "candidate_mean": float(cand.mean()),
                    "mean_difference_candidate_minus_baseline": float(diff.mean()),
                    "bootstrap_ci_low": ci_low,
                    "bootstrap_ci_high": ci_high,
                    "wilcoxon_p": p_value,
                    "better_direction_mean": float(better.mean()),
                    "candidate_better_fraction": float((better > 0).mean()),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    original = load_metric_frame(args.original_metrics, "original", args.case_column)
    counterfactual = load_metric_frame(args.counterfactual_metrics, "cfseg_counterfactual", args.case_column)
    patching = pd.read_csv(args.patching_metrics)

    conditions = [
        original,
        counterfactual,
        patch_condition(
            patching,
            control="matched_cf_to_original",
            region="whole",
            label="matched_whole",
            case_column=args.case_column,
        ),
        patch_condition(
            patching,
            control="reverse_original_to_cf",
            region="whole",
            label="reverse_whole",
            case_column=args.case_column,
        ),
        patch_condition(
            patching,
            control="nonmatched_cf_to_original",
            region="whole",
            label="nonmatched_whole",
            case_column=args.case_column,
        ),
        patch_condition(
            patching,
            control="matched_cf_to_original",
            region="background",
            label="matched_background",
            case_column=args.case_column,
        ),
        patch_condition(
            patching,
            control="matched_cf_to_original",
            region="boundary",
            label="matched_boundary",
            case_column=args.case_column,
        ),
        patch_condition(
            patching,
            control="matched_cf_to_original",
            region="random_like_boundary",
            label="matched_random_like_boundary",
            case_column=args.case_column,
        ),
        patch_condition(
            patching,
            control="matched_cf_to_original",
            region="random_like_background",
            label="matched_random_like_background",
            case_column=args.case_column,
        ),
    ]

    merged = conditions[0]
    for condition in conditions[1:]:
        merged = merged.merge(condition, on=args.case_column, how="inner", validate="one_to_one")

    comparisons = [
        ("original", "cfseg_counterfactual"),
        ("original", "matched_whole"),
        ("cfseg_counterfactual", "matched_whole"),
        ("original", "reverse_whole"),
        ("matched_whole", "reverse_whole"),
        ("nonmatched_whole", "matched_whole"),
        ("matched_boundary", "matched_background"),
        ("matched_random_like_boundary", "matched_boundary"),
        ("matched_random_like_background", "matched_background"),
    ]
    stats = paired_table(
        merged,
        case_column=args.case_column,
        comparisons=comparisons,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )

    mean_rows = []
    for label in sorted({column.split("__", 1)[0] for column in merged.columns if "__" in column}):
        row = {"condition": label, "n": len(merged)}
        for metric in METRICS:
            row[metric] = float(merged[f"{label}__{metric}"].mean())
        mean_rows.append(row)
    means = pd.DataFrame(mean_rows)

    merged.to_csv(args.output_dir / "paired_case_metrics.csv", index=False)
    means.to_csv(args.output_dir / "condition_means.csv", index=False)
    stats.to_csv(args.output_dir / "paired_bootstrap_wilcoxon.csv", index=False)
    print(f"Wrote statistical tables to {args.output_dir}")


if __name__ == "__main__":
    main()
