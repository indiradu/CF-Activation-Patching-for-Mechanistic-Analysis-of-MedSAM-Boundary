#!/usr/bin/env python3
"""Merge original and counterfactual MedSAM prediction columns into one manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-manifest", required=True, type=Path)
    parser.add_argument("--prediction-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--case-column", default="case_id")
    parser.add_argument("--columns", nargs="+", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base = pd.read_csv(args.base_manifest)
    pred = pd.read_csv(args.prediction_manifest)

    keep = [args.case_column, *args.columns]
    missing = [column for column in keep if column not in pred.columns]
    if missing:
        raise ValueError(f"Prediction manifest missing columns: {missing}")

    merged = base.merge(pred[keep], on=args.case_column, how="left", validate="one_to_one")
    for column in args.columns:
        if merged[column].isna().any():
            missing_cases = merged.loc[merged[column].isna(), args.case_column].tolist()
            raise ValueError(f"Missing {column} for cases: {missing_cases[:10]}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output, index=False)
    print(f"Wrote merged manifest: {args.output}")


if __name__ == "__main__":
    main()
