#!/usr/bin/env python3
"""Decode CheXMask ChestX-Ray8 lung masks for cases in a manifest."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--chexmask-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--output-manifest", required=True, type=Path)
    parser.add_argument("--image-id-column", default="image_id")
    parser.add_argument("--chunksize", type=int, default=5000)
    parser.add_argument("--rle-index-base", choices=[0, 1], type=int, default=0)
    parser.add_argument("--rle-order", choices=["C", "F"], default="C")
    return parser.parse_args()


def decode_rle(
    rle: object,
    *,
    height: int,
    width: int,
    index_base: int = 0,
    order: str = "C",
) -> np.ndarray:
    if rle is None or (isinstance(rle, float) and math.isnan(rle)):
        return np.zeros((height, width), dtype=bool)
    parts = str(rle).strip().split()
    if not parts:
        return np.zeros((height, width), dtype=bool)
    if len(parts) % 2 != 0:
        raise ValueError("RLE must contain start/length pairs")

    values = np.asarray(parts, dtype=np.int64)
    starts = values[0::2]
    lengths = values[1::2]

    flat = np.zeros(height * width, dtype=np.uint8)
    for start, length in zip(starts, lengths):
        start_index = max(int(start) - index_base, 0)
        end_index = min(start_index + int(length), flat.size)
        flat[start_index:end_index] = 1
    if order == "F":
        return flat.reshape((width, height)).T.astype(bool)
    return flat.reshape((height, width)).astype(bool)


def save_mask(mask: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8) * 255).save(path)


def main() -> None:
    args = parse_args()
    manifest = pd.read_csv(args.manifest)
    wanted = set(manifest[args.image_id_column].astype(str))
    if not wanted:
        raise SystemExit("Manifest contains no image IDs")

    matches: dict[str, Path] = {}
    usecols = ["Image Index", "Left Lung", "Right Lung", "Height", "Width"]
    for chunk in pd.read_csv(args.chexmask_csv, usecols=usecols, chunksize=args.chunksize):
        chunk["Image Index"] = chunk["Image Index"].astype(str)
        chunk = chunk[chunk["Image Index"].isin(wanted.difference(matches))]
        if chunk.empty:
            continue

        for _, row in chunk.iterrows():
            image_id = str(row["Image Index"])
            left_rle = row["Left Lung"]
            right_rle = row["Right Lung"]
            height = int(row["Height"])
            width = int(row["Width"])
            left = decode_rle(
                left_rle,
                height=height,
                width=width,
                index_base=args.rle_index_base,
                order=args.rle_order,
            )
            right = decode_rle(
                right_rle,
                height=height,
                width=width,
                index_base=args.rle_index_base,
                order=args.rle_order,
            )
            mask = np.logical_or(left, right)
            output_path = args.output_dir / f"{Path(image_id).stem}_lung.png"
            save_mask(mask, output_path)
            matches[image_id] = output_path

        print(f"Decoded {len(matches)} / {len(wanted)} masks")
        if len(matches) == len(wanted):
            break

    missing = wanted.difference(matches)
    if missing:
        preview = ", ".join(sorted(missing)[:10])
        raise SystemExit(f"Missing CheXMask rows for {len(missing)} image IDs: {preview}")

    manifest["lung_mask_path"] = manifest[args.image_id_column].astype(str).map(lambda image_id: str(matches[image_id]))
    manifest["mask_source"] = "chexmask_chestxray8_original_resolution"
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(args.output_manifest, index=False)
    print(f"Wrote manifest with masks to {args.output_manifest}")


if __name__ == "__main__":
    main()
