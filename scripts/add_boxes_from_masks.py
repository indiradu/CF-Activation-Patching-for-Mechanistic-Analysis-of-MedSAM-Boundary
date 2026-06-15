#!/usr/bin/env python3
"""Add MedSAM prompt boxes derived from binary lung masks."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from counterfactual_medsam.manifest import add_boxes_from_masks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--mask-column", default="lung_mask_path")
    parser.add_argument("--box-column", default="box_xyxy")
    parser.add_argument("--margin", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.manifest)
    df = add_boxes_from_masks(
        df,
        mask_column=args.mask_column,
        box_column=args.box_column,
        margin=args.margin,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Wrote {len(df)} rows with boxes to {args.output}")


if __name__ == "__main__":
    main()
