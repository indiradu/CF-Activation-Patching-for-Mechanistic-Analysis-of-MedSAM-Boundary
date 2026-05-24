#!/usr/bin/env python3
"""Summarize MedSAM activation-patching metrics into paper-ready CSV tables."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


METRIC_COLUMNS = ["dice", "iou", "boundary_f1", "hd95", "area_error"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--patching-metrics", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--best-metric", default="boundary_f1", choices=METRIC_COLUMNS)
    return parser.parse_args()


def metric_summary(df: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    return (
        df.groupby(group_columns, dropna=False)[METRIC_COLUMNS]
        .mean()
        .reset_index()
        .sort_values(["boundary_f1", "dice"], ascending=[False, False])
    )


def best_by_case(df: pd.DataFrame, group_columns: list[str], best_metric: str) -> pd.DataFrame:
    sort_ascending = best_metric in {"hd95", "area_error"}
    order = df.sort_values(best_metric, ascending=sort_ascending)
    return order.groupby(group_columns, as_index=False, dropna=False).head(1)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.patching_metrics)
    if "control" not in df.columns:
        df = df.copy()
        df["control"] = "matched_cf_to_original"

    metric_summary(df, ["control"]).to_csv(args.output_dir / "summary_by_control.csv", index=False)
    metric_summary(df, ["control", "region"]).to_csv(
        args.output_dir / "summary_by_control_region.csv",
        index=False,
    )
    metric_summary(df, ["control", "layer", "region"]).to_csv(
        args.output_dir / "summary_by_control_layer_region.csv",
        index=False,
    )
    best_by_case(df, ["case_id", "control"], args.best_metric).to_csv(
        args.output_dir / "best_patch_by_case_control.csv",
        index=False,
    )

    print(f"Wrote summaries to {args.output_dir}")


if __name__ == "__main__":
    main()
