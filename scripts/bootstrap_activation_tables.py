#!/usr/bin/env python3
"""Bootstrap paired statistics for original/counterfactual activation controls."""

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
    parser.add_argument("--counterfactual-label", default="counterfactual")
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
    return df.rename(columns={metric: f"{label}__{metric}" for metric in METRICS})


def patch_condition(
    df: pd.DataFrame,
    *,
    control: str,
    region: str,
    label: str,
    case_column: str,
    layer: str | None = None,
) -> pd.DataFrame | None:
    subset = df[(df["control"] == control) & (df["region"] == region)].copy()
    if layer is not None:
        subset = subset[subset["layer"] == layer].copy()
    if subset.empty:
        return None
    subset = subset.groupby(case_column, as_index=False)[METRICS].mean()
    return subset.rename(columns={metric: f"{label}__{metric}" for metric in METRICS})


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator, samples: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    draws = rng.choice(values, size=(samples, len(values)), replace=True)
    means = draws.mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired_table(
    df: pd.DataFrame,
    *,
    labels: set[str],
    case_column: str,
    comparisons: list[tuple[str, str]],
    bootstrap_samples: int,
    seed: int,
) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(seed)
    for baseline, candidate in comparisons:
        if baseline not in labels or candidate not in labels:
            continue
        for metric in METRICS:
            base = df[f"{baseline}__{metric}"].to_numpy(dtype=float)
            cand = df[f"{candidate}__{metric}"].to_numpy(dtype=float)
            diff = cand - base
            ci_low, ci_high = bootstrap_ci(diff, rng, bootstrap_samples)
            try:
                p_value = float(wilcoxon(diff, zero_method="wilcox", alternative="two-sided").pvalue)
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
                    "candidate_better_fraction": float((better > 0).mean()),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    patching = pd.read_csv(args.patching_metrics)
    if "control" not in patching.columns:
        patching = patching.copy()
        patching["control"] = "matched_cf_to_original"

    conditions: list[tuple[str, pd.DataFrame]] = [
        ("original", load_metric_frame(args.original_metrics, "original", args.case_column)),
        (
            args.counterfactual_label,
            load_metric_frame(args.counterfactual_metrics, args.counterfactual_label, args.case_column),
        ),
    ]

    requested_patch_conditions = [
        ("matched_cf_to_original", "whole", "matched_whole"),
        ("reverse_original_to_cf", "whole", "reverse_whole"),
        ("nonmatched_cf_to_original", "whole", "nonmatched_whole"),
        ("matched_cf_to_original", "background", "matched_background"),
        ("matched_cf_to_original", "boundary", "matched_boundary"),
        ("matched_cf_to_original", "interior", "matched_interior"),
        ("matched_cf_to_original", "random_like_boundary", "matched_random_like_boundary"),
        ("matched_cf_to_original", "random_like_background", "matched_random_like_background"),
    ]
    for control, region, label in requested_patch_conditions:
        condition = patch_condition(
            patching,
            control=control,
            region=region,
            label=label,
            case_column=args.case_column,
        )
        if condition is not None:
            conditions.append((label, condition))

    merged = conditions[0][1]
    labels = {conditions[0][0]}
    for label, condition in conditions[1:]:
        merged = merged.merge(condition, on=args.case_column, how="inner", validate="one_to_one")
        labels.add(label)
    if merged.empty:
        raise ValueError("No paired cases remain after merging conditions")

    comparisons = [
        ("original", args.counterfactual_label),
        ("original", "matched_whole"),
        (args.counterfactual_label, "matched_whole"),
        ("original", "reverse_whole"),
        ("matched_whole", "reverse_whole"),
        ("nonmatched_whole", "matched_whole"),
        ("matched_boundary", "matched_background"),
        ("matched_interior", "matched_background"),
        ("matched_random_like_boundary", "matched_boundary"),
        ("matched_random_like_background", "matched_background"),
    ]
    stats = paired_table(
        merged,
        labels=labels,
        case_column=args.case_column,
        comparisons=comparisons,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )

    mean_rows = []
    for label in sorted(labels):
        row = {"condition": label, "n": int(merged.shape[0])}
        for metric in METRICS:
            row[metric] = float(merged[f"{label}__{metric}"].mean())
        mean_rows.append(row)

    merged.to_csv(args.output_dir / "paired_case_metrics.csv", index=False)
    pd.DataFrame(mean_rows).to_csv(args.output_dir / "condition_means.csv", index=False)
    stats.to_csv(args.output_dir / "paired_bootstrap_wilcoxon.csv", index=False)
    print(f"Wrote activation bootstrap tables to {args.output_dir} using {merged.shape[0]} paired cases")


if __name__ == "__main__":
    main()
