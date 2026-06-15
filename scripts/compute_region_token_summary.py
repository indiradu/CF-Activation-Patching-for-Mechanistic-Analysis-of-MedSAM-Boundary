#!/usr/bin/env python3
"""Summarize full-resolution patch regions after resizing to activation tokens."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from counterfactual_medsam.regions import probing_region, read_mask, resize_mask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--reference-column", default="lung_mask_path")
    parser.add_argument("--boundary-width", type=int, default=32)
    parser.add_argument("--feature-height", type=int, default=64)
    parser.add_argument("--feature-width", type=int, default=64)
    parser.add_argument("--regions", default="whole,boundary,background,interior")
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.manifest)
    regions = [region.strip() for region in args.regions.split(",") if region.strip()]
    feature_shape = (args.feature_height, args.feature_width)

    rows = []
    for _, row in df.iterrows():
        reference = read_mask(Path(str(row[args.reference_column])))
        for region in regions:
            full_region = probing_region(reference, region, boundary_width=args.boundary_width)
            token_region = resize_mask(full_region, feature_shape)
            rows.append(
                {
                    "region": region,
                    "tokens": int(token_region.sum()),
                    "token_fraction": float(token_region.mean()),
                }
            )

    summary = (
        pd.DataFrame(rows)
        .groupby("region")
        .agg(
            tokens_mean=("tokens", "mean"),
            tokens_sd=("tokens", "std"),
            token_fraction_mean=("token_fraction", "mean"),
            token_fraction_sd=("token_fraction", "std"),
        )
        .reset_index()
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output, index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
