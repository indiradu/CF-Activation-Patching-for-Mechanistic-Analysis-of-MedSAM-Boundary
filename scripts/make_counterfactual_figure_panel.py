#!/usr/bin/env python3
"""Render paper-style panels for original/counterfactual MedSAM comparisons."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from counterfactual_medsam.manifest import parse_box_xyxy


@dataclass(frozen=True)
class Condition:
    label: str
    manifest: Path
    image_column: str
    prediction_column: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", action="append", required=True, help="LABEL:MANIFEST:IMAGE_COLUMN:PRED_COLUMN")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--case-ids", help="Comma-separated case ids. Defaults to the first cases in the first manifest.")
    parser.add_argument("--case-column", default="case_id")
    parser.add_argument("--mask-column", default="lung_mask_path")
    parser.add_argument("--box-column", default="box_xyxy")
    parser.add_argument("--max-cases", type=int, default=4)
    parser.add_argument("--tile-width", type=int, default=300)
    parser.add_argument("--header-height", type=int, default=34)
    parser.add_argument("--caption-height", type=int, default=32)
    return parser.parse_args()


def parse_condition(raw: str) -> Condition:
    parts = raw.split(":")
    if len(parts) != 4:
        raise ValueError(
            "--condition must have format LABEL:MANIFEST:IMAGE_COLUMN:PRED_COLUMN; "
            f"got {raw!r}"
        )
    label, manifest, image_column, prediction_column = parts
    return Condition(label=label, manifest=Path(manifest), image_column=image_column, prediction_column=prediction_column)


def resolve_path(path_value: object, project_root: Path) -> Path:
    path = Path(str(path_value))
    if path.is_absolute() and path.exists():
        return path
    if not path.is_absolute():
        candidate = project_root / path
        if candidate.exists():
            return candidate
    if path.is_absolute():
        marker = "miccai_workshop/"
        text = str(path)
        if marker in text:
            candidate = project_root / text.split(marker, 1)[1]
            if candidate.exists():
                return candidate
    return path


def mask_boundary(mask: np.ndarray) -> np.ndarray:
    mask = mask.astype(bool)
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    up = padded[:-2, 1:-1]
    down = padded[2:, 1:-1]
    left = padded[1:-1, :-2]
    right = padded[1:-1, 2:]
    eroded = mask & up & down & left & right
    return mask & ~eroded


def overlay_mask(image: Image.Image, reference: Image.Image, prediction: Image.Image | None) -> Image.Image:
    arr = np.asarray(image.convert("RGB")).astype(np.float32)
    ref = np.asarray(reference.convert("L")) > 0
    ref_edge = mask_boundary(ref)

    if prediction is not None:
        pred = np.asarray(prediction.convert("L")) > 0
        pred_edge = mask_boundary(pred)
        cyan = np.array([0, 210, 255], dtype=np.float32)
        arr[pred] = 0.74 * arr[pred] + 0.26 * cyan
        arr[pred_edge] = np.array([0, 235, 255], dtype=np.float32)

    orange = np.array([255, 132, 38], dtype=np.float32)
    arr[ref_edge] = orange
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="RGB")


def load_font(size: int = 14) -> ImageFont.ImageFont:
    for candidate in [
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def make_tile(
    row: pd.Series,
    *,
    condition: Condition,
    project_root: Path,
    tile_width: int,
    caption_height: int,
    box_column: str,
    case_column: str,
    mask_column: str,
) -> Image.Image:
    image = Image.open(resolve_path(row[condition.image_column], project_root)).convert("RGB")
    reference = Image.open(resolve_path(row[mask_column], project_root)).convert("L")
    prediction_path = row.get(condition.prediction_column, "")
    prediction = None
    if pd.notna(prediction_path) and str(prediction_path).strip():
        prediction = Image.open(resolve_path(prediction_path, project_root)).convert("L")

    if image.size != reference.size:
        raise ValueError(f"Image/reference size mismatch for {row[case_column]}: {image.size} vs {reference.size}")
    if prediction is not None and image.size != prediction.size:
        raise ValueError(f"Image/prediction size mismatch for {row[case_column]}: {image.size} vs {prediction.size}")

    composed = overlay_mask(image, reference, prediction).convert("RGBA")
    draw = ImageDraw.Draw(composed)
    box = parse_box_xyxy(row[box_column])
    draw.rectangle(box, outline=(255, 230, 0, 255), width=max(4, image.width // 512))

    scale = tile_width / image.width
    tile_height = int(image.height * scale)
    composed = composed.resize((tile_width, tile_height), Image.Resampling.LANCZOS).convert("RGB")

    tile = Image.new("RGB", (tile_width, tile_height + caption_height), "white")
    tile.paste(composed, (0, 0))
    draw = ImageDraw.Draw(tile)
    font = load_font(13)
    draw.text((8, tile_height + 7), str(row[case_column])[:36], fill=(20, 20, 20), font=font)
    return tile


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    conditions = [parse_condition(raw) for raw in args.condition]

    frames = []
    for condition in conditions:
        manifest_path = resolve_path(condition.manifest, project_root)
        df = pd.read_csv(manifest_path)
        df = df.drop_duplicates(args.case_column).set_index(args.case_column, drop=False)
        frames.append((condition, df))

    if args.case_ids:
        case_ids = [case_id.strip() for case_id in args.case_ids.split(",") if case_id.strip()]
    else:
        case_ids = list(frames[0][1].index[: args.max_cases])
    case_ids = case_ids[: args.max_cases]

    missing = {
        condition.label: [case_id for case_id in case_ids if case_id not in df.index]
        for condition, df in frames
    }
    missing = {label: values for label, values in missing.items() if values}
    if missing:
        raise ValueError(f"Some requested cases are missing from condition manifests: {missing}")

    font = load_font(15)
    title_font = load_font(17)
    sample_tile = make_tile(
        frames[0][1].loc[case_ids[0]],
        condition=frames[0][0],
        project_root=project_root,
        tile_width=args.tile_width,
        caption_height=args.caption_height,
        box_column=args.box_column,
        case_column=args.case_column,
        mask_column=args.mask_column,
    )
    tile_height = sample_tile.height
    width = args.tile_width * len(frames)
    height = args.header_height + tile_height * len(case_ids)
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)

    for col_idx, (condition, _) in enumerate(frames):
        x = col_idx * args.tile_width
        draw.rectangle((x, 0, x + args.tile_width, args.header_height), fill=(245, 245, 242))
        draw.text((x + 8, 8), condition.label, fill=(18, 18, 18), font=title_font)

    for row_idx, case_id in enumerate(case_ids):
        y = args.header_height + row_idx * tile_height
        for col_idx, (condition, df) in enumerate(frames):
            tile = make_tile(
                df.loc[case_id],
                condition=condition,
                project_root=project_root,
                tile_width=args.tile_width,
                caption_height=args.caption_height,
                box_column=args.box_column,
                case_column=args.case_column,
                mask_column=args.mask_column,
            )
            sheet.paste(tile, (col_idx * args.tile_width, y))

    legend_height = 28
    final = Image.new("RGB", (width, height + legend_height), "white")
    final.paste(sheet, (0, 0))
    draw = ImageDraw.Draw(final)
    draw.text(
        (8, height + 7),
        "Orange: CheXMask reference boundary. Cyan: MedSAM prediction. Yellow: prompt box.",
        fill=(35, 35, 35),
        font=font,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    final.save(args.output)
    print(f"Wrote figure panel to {args.output}")


if __name__ == "__main__":
    main()
