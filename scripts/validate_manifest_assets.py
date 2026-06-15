#!/usr/bin/env python3
"""Validate image, mask, and box assets referenced by an experiment manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from PIL import Image

from counterfactual_medsam.manifest import parse_box_xyxy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--image-column", default="image_path")
    parser.add_argument("--mask-column", default="lung_mask_path")
    parser.add_argument("--box-column", default="box_xyxy")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.manifest)
    problems = []

    for row_idx, row in df.iterrows():
        image_path = Path(str(row[args.image_column]))
        mask_path = Path(str(row[args.mask_column]))
        if not image_path.exists():
            problems.append(f"row {row_idx}: missing image {image_path}")
            continue
        if not mask_path.exists():
            problems.append(f"row {row_idx}: missing mask {mask_path}")
            continue

        with Image.open(image_path) as image, Image.open(mask_path) as mask:
            image_size = image.size
            mask_size = mask.size

        if image_size != mask_size:
            problems.append(f"row {row_idx}: image/mask size mismatch {image_size} vs {mask_size}")

        try:
            x1, y1, x2, y2 = parse_box_xyxy(row[args.box_column])
        except Exception as exc:  # noqa: BLE001 - validation should aggregate malformed rows.
            problems.append(f"row {row_idx}: invalid box {row[args.box_column]!r}: {exc}")
            continue

        width, height = image_size
        if not (0 <= x1 < x2 < width and 0 <= y1 < y2 < height):
            problems.append(f"row {row_idx}: box outside image {x1,y1,x2,y2} for size {image_size}")

    print(f"Rows: {len(df)}")
    print(f"Unique images: {df[args.image_column].nunique()}")
    print(f"Unique masks: {df[args.mask_column].nunique()}")

    if problems:
        print("Problems:")
        for problem in problems[:20]:
            print(f"- {problem}")
        if len(problems) > 20:
            print(f"- ... {len(problems) - 20} more")
        raise SystemExit(1)

    print("Manifest assets validated successfully.")


if __name__ == "__main__":
    main()
