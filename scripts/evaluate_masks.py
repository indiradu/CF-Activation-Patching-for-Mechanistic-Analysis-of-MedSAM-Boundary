#!/usr/bin/env python3
"""Evaluate predicted masks against reference lung masks."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from counterfactual_medsam.metrics import evaluate_mask_pair


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--prediction-column", required=True)
    parser.add_argument("--reference-column", default="lung_mask_path")
    parser.add_argument("--case-column", default="case_id")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--boundary-tolerance", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.manifest)

    records = []
    for _, row in df.iterrows():
        metrics = evaluate_mask_pair(
            prediction_path=Path(str(row[args.prediction_column])),
            reference_path=Path(str(row[args.reference_column])),
            boundary_tolerance=args.boundary_tolerance,
        )
        metrics[args.case_column] = row[args.case_column]
        metrics["prediction_path"] = row[args.prediction_column]
        metrics["reference_path"] = row[args.reference_column]
        records.append(metrics)

    out = pd.DataFrame(records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    numeric = out.select_dtypes(include="number")
    print(numeric.describe().to_string())
    print(f"Wrote metrics to {args.output}")


if __name__ == "__main__":
    main()
