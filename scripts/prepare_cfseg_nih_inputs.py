#!/usr/bin/env python3
"""Materialize NIH manifest rows in the CF-Seg causal-gen input layout."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--case-column", default="case_id")
    parser.add_argument("--image-column", default="image_path")
    parser.add_argument("--mask-column", default="lung_mask_path")
    parser.add_argument("--view-column", default="view_position")
    parser.add_argument("--subject-column", default="subject_id")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--split", default="test")
    parser.add_argument("--default-age", type=float, default=60.0)
    parser.add_argument("--default-race", choices=["White", "Asian", "Black"], default="White")
    parser.add_argument("--default-sex", choices=["Male", "Female"], default="Female")
    parser.add_argument("--duplicate-all-splits", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


RACE_TO_LABEL = {"White": 0, "Asian": 1, "Black": 2}
SEX_TO_LABEL = {"Male": 0, "Female": 1}
VIEW_TO_LABEL = {"AP": 0, "PA": 1}


def _resize_image(src: Path, dst: Path, size: int, mode: str) -> None:
    if not src.exists():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    image = Image.open(src).convert(mode)
    resample = Image.Resampling.LANCZOS if mode == "L" else Image.Resampling.NEAREST
    image = image.resize((size, size), resample)
    image.save(dst)


def _view_label(value: object) -> tuple[str, int]:
    view = str(value).upper().strip()
    if view not in VIEW_TO_LABEL:
        view = "PA"
    return view, VIEW_TO_LABEL[view]


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.manifest)

    image_dir = args.output_root / "CheXMask_files_preprocessed"
    label_dir = args.output_root / "CheXMask_segmentation_preprocessed"
    csv_dir = args.output_root / "mimic_csv"

    rows = []
    for idx, row in df.iterrows():
        case_id = str(row[args.case_column])
        image_src = Path(str(row[args.image_column]))
        mask_src = Path(str(row[args.mask_column]))
        view, view_label = _view_label(row.get(args.view_column, "PA"))

        # CF-Seg's dataloader appends .jpg/.png to the dicom_id field.
        _resize_image(image_src, image_dir / f"{case_id}.jpg", args.image_size, "L")
        _resize_image(mask_src, label_dir / f"{case_id}.png", args.image_size, "RGB")

        subject = row.get(args.subject_column, idx)
        rows.append(
            {
                "dicom_id": case_id,
                "subject_id": subject,
                "study_id": 0,
                "age": args.default_age,
                "split": args.split,
                "ViewPosition": view,
                "ViewPosition_label": view_label,
                "race": args.default_race,
                "race_label": RACE_TO_LABEL[args.default_race],
                "sex": args.default_sex,
                "sex_label": SEX_TO_LABEL[args.default_sex],
                "disease": "Pleural Effusion",
                "disease_label": 1,
                "Dice RCA (Mean)": 1.0,
                "Dice RCA (Max)": 1.0,
            }
        )

    csv_dir.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame(rows)
    split_csv = csv_dir / f"{args.split}_pe_only.csv"
    out.to_csv(split_csv, index=False)
    if args.duplicate_all_splits:
        out.assign(split="train").to_csv(csv_dir / "train_pe_only.csv", index=False)
        out.assign(split="valid").to_csv(csv_dir / "valid_pe_only.csv", index=False)
        out.assign(split="test").to_csv(csv_dir / "test_pe_only.csv", index=False)

    print(f"Wrote {len(out)} CF-Seg rows to {split_csv}")
    print(f"Images: {image_dir}")
    print(f"Masks: {label_dir}")


if __name__ == "__main__":
    main()
