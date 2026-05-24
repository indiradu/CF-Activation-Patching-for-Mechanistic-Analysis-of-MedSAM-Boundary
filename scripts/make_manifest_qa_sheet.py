#!/usr/bin/env python3
"""Create a visual QA sheet for image, reference mask, prediction, and box rows."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from counterfactual_medsam.manifest import parse_box_xyxy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--image-column", default="image_path")
    parser.add_argument("--mask-column", default="lung_mask_path")
    parser.add_argument("--prediction-column")
    parser.add_argument("--box-column", default="box_xyxy")
    parser.add_argument("--case-column", default="case_id")
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--tile-width", type=int, default=360)
    parser.add_argument("--columns", type=int, default=4)
    return parser.parse_args()


def make_tile(row: pd.Series, args: argparse.Namespace) -> Image.Image:
    image = Image.open(str(row[args.image_column])).convert("RGB")
    mask = Image.open(str(row[args.mask_column])).convert("L")
    if image.size != mask.size:
        raise ValueError(f"Image/mask size mismatch for {row[args.case_column]}: {image.size} vs {mask.size}")

    width, height = image.size
    image_arr = np.asarray(image).copy()
    mask_arr = np.asarray(mask) > 0
    red = np.array([255, 64, 0], dtype=np.float32)
    image_arr[mask_arr] = (0.72 * image_arr[mask_arr].astype(np.float32) + 0.28 * red).astype(np.uint8)

    if args.prediction_column and pd.notna(row.get(args.prediction_column, "")):
        prediction_path = str(row[args.prediction_column]).strip()
        if prediction_path:
            prediction = Image.open(prediction_path).convert("L")
            if image.size != prediction.size:
                raise ValueError(
                    f"Image/prediction size mismatch for {row[args.case_column]}: "
                    f"{image.size} vs {prediction.size}"
                )
            prediction_arr = np.asarray(prediction) > 0
            cyan = np.array([0, 210, 255], dtype=np.float32)
            image_arr[prediction_arr] = (
                0.68 * image_arr[prediction_arr].astype(np.float32) + 0.32 * cyan
            ).astype(np.uint8)
    composed = Image.fromarray(image_arr, mode="RGB").convert("RGBA")
    draw = ImageDraw.Draw(composed)
    box = parse_box_xyxy(row[args.box_column])
    draw.rectangle(box, outline=(255, 230, 0, 255), width=max(4, width // 512))

    scale = args.tile_width / width
    tile_height = int(height * scale)
    composed = composed.resize((args.tile_width, tile_height), Image.Resampling.LANCZOS)

    caption_height = 28
    tile = Image.new("RGB", (args.tile_width, tile_height + caption_height), "white")
    tile.paste(composed.convert("RGB"), (0, 0))
    draw = ImageDraw.Draw(tile)
    font = ImageFont.load_default()
    draw.text((6, tile_height + 7), str(row[args.case_column])[:54], fill=(20, 20, 20), font=font)
    return tile


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.manifest).head(args.limit)
    tiles = [make_tile(row, args) for _, row in df.iterrows()]
    if not tiles:
        raise SystemExit("No rows to render")

    columns = max(1, args.columns)
    rows = (len(tiles) + columns - 1) // columns
    tile_width = max(tile.width for tile in tiles)
    tile_height = max(tile.height for tile in tiles)
    sheet = Image.new("RGB", (columns * tile_width, rows * tile_height), "white")

    for idx, tile in enumerate(tiles):
        x = (idx % columns) * tile_width
        y = (idx // columns) * tile_height
        sheet.paste(tile, (x, y))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.output)
    print(f"Wrote QA sheet to {args.output}")


if __name__ == "__main__":
    main()
