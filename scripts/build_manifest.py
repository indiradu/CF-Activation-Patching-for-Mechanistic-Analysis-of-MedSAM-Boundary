#!/usr/bin/env python3
"""Build the first MIMIC-CXR pleural-effusion manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

from counterfactual_medsam.manifest import build_mimic_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mimic-root", required=True, type=Path)
    parser.add_argument("--metadata-csv", required=True, type=Path)
    parser.add_argument("--labels-csv", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--split-csv", type=Path)
    parser.add_argument("--mask-index", type=Path)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--cohort", choices=["pleural_effusion", "no_finding", "both"], default="pleural_effusion")
    parser.add_argument("--views", nargs="+", default=["AP", "PA"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_mimic_manifest(
        mimic_root=args.mimic_root,
        metadata_csv=args.metadata_csv,
        labels_csv=args.labels_csv,
        output_csv=args.output,
        split_csv=args.split_csv,
        mask_index_csv=args.mask_index,
        cohort=args.cohort,
        views=tuple(args.views),
        limit=args.limit,
    )
    print(f"Wrote {len(manifest)} rows to {args.output}")


if __name__ == "__main__":
    main()

