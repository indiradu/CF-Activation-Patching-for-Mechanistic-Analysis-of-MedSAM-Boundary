#!/usr/bin/env python3
"""Copy manifest assets into a portable runtime directory and rewrite paths."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-manifest", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--case-column", default="case_id")
    parser.add_argument("--image-column", default="image_path")
    parser.add_argument("--mask-column", default="lung_mask_path")
    return parser.parse_args()


def _copy_one(src: Path, dst: Path) -> Path:
    if not src.exists():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists() or src.stat().st_size != dst.stat().st_size:
        shutil.copy2(src, dst)
    return dst


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.manifest)
    rows = []

    for row in df.to_dict(orient="records"):
        case_id = str(row[args.case_column])

        image_src = Path(str(row[args.image_column]))
        image_dst = args.output_root / "images" / f"{case_id}{image_src.suffix}"
        row[args.image_column] = str(_copy_one(image_src, image_dst))

        mask_value = row.get(args.mask_column, "")
        if pd.notna(mask_value) and str(mask_value).strip():
            mask_src = Path(str(mask_value))
            mask_dst = args.output_root / "masks" / f"{case_id}{mask_src.suffix}"
            row[args.mask_column] = str(_copy_one(mask_src, mask_dst))

        rows.append(row)

    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output_manifest, index=False)
    print(f"Wrote runtime manifest: {args.output_manifest}")
    print(f"Materialized {len(rows)} rows under: {args.output_root}")


if __name__ == "__main__":
    main()
