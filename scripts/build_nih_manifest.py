#!/usr/bin/env python3
"""Build a NIH ChestX-ray14 manifest for the no-MIMIC MVP path."""

from __future__ import annotations

import argparse
from pathlib import Path

from counterfactual_medsam.manifest import build_nih_chestxray14_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-root", required=True, type=Path)
    parser.add_argument("--labels-csv", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--mask-index", type=Path)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--cohort", choices=["effusion", "no_finding", "both"], default="effusion")
    parser.add_argument("--views", nargs="+", default=["AP", "PA"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_nih_chestxray14_manifest(
        image_root=args.image_root,
        labels_csv=args.labels_csv,
        output_csv=args.output,
        mask_index_csv=args.mask_index,
        cohort=args.cohort,
        views=tuple(args.views),
        limit=args.limit,
    )
    print(f"Wrote {len(manifest)} rows to {args.output}")


if __name__ == "__main__":
    main()
