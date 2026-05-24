#!/usr/bin/env python3
"""Import CF-Seg generated counterfactuals into the project manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--cfseg-dir", required=True, type=Path)
    parser.add_argument("--output-manifest", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--intervention-dir", default="pe_finding_0")
    parser.add_argument("--case-column", default="case_id")
    parser.add_argument("--image-column", default="image_path")
    parser.add_argument("--counterfactual-column", default="counterfactual_path")
    parser.add_argument("--source-column", default="counterfactual_source")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.manifest)
    rows = []

    for row in df.to_dict(orient="records"):
        case_id = str(row[args.case_column])
        cf_src = args.cfseg_dir / args.intervention_dir / f"{case_id}_cf.png"
        if not cf_src.exists():
            raise FileNotFoundError(cf_src)

        original = Image.open(str(row[args.image_column])).convert("L")
        cf = Image.open(cf_src).convert("L").resize(original.size, Image.Resampling.BICUBIC)

        cf_dst = args.output_root / f"{case_id}_cfseg_{args.intervention_dir}.png"
        cf_dst.parent.mkdir(parents=True, exist_ok=True)
        cf.save(cf_dst)

        row[args.counterfactual_column] = str(cf_dst)
        row[args.source_column] = f"cfseg:{args.intervention_dir}"
        rows.append(row)

    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output_manifest, index=False)
    print(f"Wrote counterfactual manifest: {args.output_manifest}")
    print(f"Imported {len(rows)} counterfactuals into: {args.output_root}")


if __name__ == "__main__":
    main()
