#!/usr/bin/env python3
"""Compare multiple case-level metric CSVs with paired bootstrap statistics."""

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
    parser.add_argument("--metric-table", action="append", required=True, help="LABEL=CSV_PATH")
    parser.add_argument("--comparison", action="append", required=True, help="BASELINE:CANDIDATE")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--case-column", default="case_id")
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def parse_label_path(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        raise ValueError(f"Expected LABEL=CSV_PATH, got {raw!r}")
    label, path = raw.split("=", 1)
    if not label:
        raise ValueError(f"Empty label in {raw!r}")
    return label, Path(path)


def load_table(label: str, path: Path, case_column: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    keep = [case_column, *METRICS]
    missing = [column for column in keep if column not in df.columns]
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")
    df = df[keep].copy()
    df = df.rename(columns={metric: f"{label}__{metric}" for metric in METRICS})
    return df


def parse_comparison(raw: str) -> tuple[str, str]:
    parts = raw.split(":")
    if len(parts) != 2:
        raise ValueError(f"Expected BASELINE:CANDIDATE, got {raw!r}")
    return parts[0], parts[1]


def bootstrap_ci(values: np.ndarray, *, rng: np.random.Generator, samples: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    draws = rng.choice(values, size=(samples, len(values)), replace=True)
    means = draws.mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    label_paths = [parse_label_path(raw) for raw in args.metric_table]
    labels = [label for label, _ in label_paths]
    tables = [load_table(label, path, args.case_column) for label, path in label_paths]

    merged = tables[0]
    for table in tables[1:]:
        merged = merged.merge(table, on=args.case_column, how="inner", validate="one_to_one")
    if merged.empty:
        raise ValueError("No overlapping case ids across metric tables")

    mean_rows = []
    for label in labels:
        row = {"condition": label, "n": int(merged.shape[0])}
        for metric in METRICS:
            row[metric] = float(merged[f"{label}__{metric}"].mean())
        mean_rows.append(row)

    rng = np.random.default_rng(args.seed)
    stats_rows = []
    for baseline, candidate in [parse_comparison(raw) for raw in args.comparison]:
        if baseline not in labels or candidate not in labels:
            raise ValueError(f"Unknown comparison labels: {baseline}:{candidate}; available labels are {labels}")
        for metric in METRICS:
            base = merged[f"{baseline}__{metric}"].to_numpy(dtype=float)
            cand = merged[f"{candidate}__{metric}"].to_numpy(dtype=float)
            diff = cand - base
            ci_low, ci_high = bootstrap_ci(diff, rng=rng, samples=args.bootstrap_samples)
            try:
                p_value = float(wilcoxon(diff, zero_method="wilcox", alternative="two-sided").pvalue)
            except ValueError:
                p_value = float("nan")
            better = -diff if metric in LOWER_IS_BETTER else diff
            stats_rows.append(
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

    merged.to_csv(args.output_dir / "paired_case_metrics.csv", index=False)
    pd.DataFrame(mean_rows).to_csv(args.output_dir / "condition_means.csv", index=False)
    pd.DataFrame(stats_rows).to_csv(args.output_dir / "paired_bootstrap_wilcoxon.csv", index=False)
    print(f"Wrote comparison tables to {args.output_dir} using {merged.shape[0]} paired cases")


if __name__ == "__main__":
    main()
