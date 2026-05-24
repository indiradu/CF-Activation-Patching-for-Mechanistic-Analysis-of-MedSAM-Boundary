#!/usr/bin/env python3
"""Run the official MedSAM CLI over a manifest.

This wrapper keeps upstream MedSAM code in `external/MedSAM` while giving this
project a stable manifest-based interface.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
from PIL import Image

from counterfactual_medsam.manifest import parse_box_xyxy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--medsam-repo", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--image-column", default="image_path")
    parser.add_argument("--box-column", default="box_xyxy")
    parser.add_argument("--case-column", default="case_id")
    parser.add_argument("--prediction-column", default="medsam_mask_path")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.manifest)
    entrypoint = args.medsam_repo / "MedSAM_Inference.py"
    if not entrypoint.exists():
        raise FileNotFoundError(f"Expected official MedSAM CLI at {entrypoint}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for _, row in df.iterrows():
        case_id = str(row[args.case_column])
        image_path = Path(str(row[args.image_column]))
        box = parse_box_xyxy(row[args.box_column])
        output_path = args.output_dir / f"{case_id}.png"
        medsam_cli_output = args.output_dir / f"seg_{image_path.name}"

        cmd = [
            args.python,
            str(entrypoint),
            "-i",
            str(image_path),
            "-o",
            str(args.output_dir),
            "--box",
            f"[{box[0]},{box[1]},{box[2]},{box[3]}]",
            "--device",
            args.device,
            "-chk",
            str(args.checkpoint),
        ]

        print(" ".join(cmd))
        if not args.dry_run:
            env = os.environ.copy()
            env.setdefault("MPLBACKEND", "Agg")
            subprocess.run(cmd, cwd=args.medsam_repo, check=True, env=env)
            if not medsam_cli_output.exists():
                raise FileNotFoundError(f"MedSAM did not create expected output: {medsam_cli_output}")
            Image.open(medsam_cli_output).save(output_path)
            if medsam_cli_output != output_path:
                medsam_cli_output.unlink()

        updated = row.to_dict()
        updated[args.prediction_column] = str(output_path)
        rows.append(updated)

    updated_df = pd.DataFrame(rows)
    updated_manifest = args.output_dir / "manifest_with_predictions.csv"
    updated_df.to_csv(updated_manifest, index=False)
    print(f"Wrote updated manifest to {updated_manifest}")


if __name__ == "__main__":
    main()
