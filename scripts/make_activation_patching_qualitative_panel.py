#!/usr/bin/env python3
"""Render a qualitative panel for CF-Seg activation-patching controls."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--original-manifest", type=Path, default=Path("outputs/medsam/nih_effusion_50/original/manifest_with_predictions.csv"))
    parser.add_argument("--cfseg-manifest", type=Path, default=Path("outputs/medsam/nih_effusion_50/counterfactual_cfseg_pe0/manifest_with_predictions.csv"))
    parser.add_argument("--original-metrics", type=Path, default=Path("outputs/medsam/nih_effusion_50/original/metrics_vs_chexmask.csv"))
    parser.add_argument("--cfseg-metrics", type=Path, default=Path("outputs/medsam/nih_effusion_50/counterfactual_cfseg_pe0/metrics_vs_chexmask.csv"))
    parser.add_argument("--patching-metrics", type=Path, default=Path("outputs/activation_probe/nih_effusion_50_controls_bw32/patching_metrics.csv"))
    parser.add_argument("--case-ids", help="Comma-separated case ids. Defaults to representative CF-Seg Boundary-F1 improvements.")
    parser.add_argument("--max-cases", type=int, default=3)
    parser.add_argument("--layer", default="neck")
    parser.add_argument("--tile-width", type=int, default=180)
    return parser.parse_args()


def resolve(path_value: object, project_root: Path) -> Path:
    path = Path(str(path_value))
    if path.is_absolute() and path.exists():
        return path
    if not path.is_absolute():
        candidate = project_root / path
        if candidate.exists():
            return candidate
    marker = "miccai_workshop/"
    text = str(path)
    if marker in text:
        candidate = project_root / text.split(marker, 1)[1]
        if candidate.exists():
            return candidate
    return path


def load_font(size: int) -> ImageFont.ImageFont:
    for candidate in (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def parse_box(raw: object) -> tuple[int, int, int, int]:
    values = [int(round(float(part.strip()))) for part in str(raw).split(",")]
    if len(values) != 4:
        raise ValueError(f"Expected four box coordinates, got {raw!r}")
    return tuple(values)  # type: ignore[return-value]


def boundary(mask: np.ndarray) -> np.ndarray:
    mask = mask.astype(bool)
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    eroded = (
        mask
        & padded[:-2, 1:-1]
        & padded[2:, 1:-1]
        & padded[1:-1, :-2]
        & padded[1:-1, 2:]
    )
    return mask & ~eroded


def overlay(image_path: Path, reference_path: Path, prediction_path: Path, box: tuple[int, int, int, int], tile_width: int) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    reference = Image.open(reference_path).convert("L")
    prediction = Image.open(prediction_path).convert("L")
    if image.size != reference.size:
        reference = reference.resize(image.size, Image.Resampling.NEAREST)
    if image.size != prediction.size:
        prediction = prediction.resize(image.size, Image.Resampling.NEAREST)

    arr = np.asarray(image).astype(np.float32)
    pred = np.asarray(prediction) > 0
    ref_edge = boundary(np.asarray(reference) > 0)
    pred_edge = boundary(pred)

    cyan = np.array([0, 210, 255], dtype=np.float32)
    arr[pred] = 0.74 * arr[pred] + 0.26 * cyan
    arr[pred_edge] = np.array([0, 235, 255], dtype=np.float32)
    arr[ref_edge] = np.array([255, 132, 38], dtype=np.float32)

    composed = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="RGB").convert("RGBA")
    draw = ImageDraw.Draw(composed)
    draw.rectangle(box, outline=(255, 230, 0, 255), width=max(4, image.width // 512))
    tile_height = int(image.height * (tile_width / image.width))
    return composed.resize((tile_width, tile_height), Image.Resampling.LANCZOS).convert("RGB")


def select_cases(args: argparse.Namespace, project_root: Path) -> list[str]:
    if args.case_ids:
        return [case_id.strip() for case_id in args.case_ids.split(",") if case_id.strip()][: args.max_cases]

    original = pd.read_csv(resolve(args.original_metrics, project_root))[["case_id", "boundary_f1"]]
    cfseg = pd.read_csv(resolve(args.cfseg_metrics, project_root))[["case_id", "boundary_f1"]]
    merged = original.merge(cfseg, on="case_id", suffixes=("_original", "_cfseg"))
    merged["gain"] = merged["boundary_f1_cfseg"] - merged["boundary_f1_original"]
    ranked = merged.sort_values("gain", ascending=False).reset_index(drop=True)

    if len(ranked) <= args.max_cases:
        return ranked["case_id"].tolist()
    positions = [0, len(ranked) // 2, len(ranked) - 1]
    case_ids = []
    for position in positions:
        case_id = str(ranked.loc[position, "case_id"])
        if case_id not in case_ids:
            case_ids.append(case_id)
    return case_ids[: args.max_cases]


def first_patch(patching: pd.DataFrame, case_id: str, control: str, layer: str, region: str) -> str:
    rows = patching[
        (patching["case_id"] == case_id)
        & (patching["control"] == control)
        & (patching["layer"] == layer)
        & (patching["region"] == region)
    ]
    if rows.empty:
        raise ValueError(f"Missing patch mask for {case_id}, {control}, {layer}, {region}")
    return str(rows.iloc[0]["prediction_path"])


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    original = pd.read_csv(resolve(args.original_manifest, project_root)).set_index("case_id", drop=False)
    cfseg = pd.read_csv(resolve(args.cfseg_manifest, project_root)).set_index("case_id", drop=False)
    patching = pd.read_csv(resolve(args.patching_metrics, project_root))
    case_ids = select_cases(args, project_root)

    columns = [
        "Original",
        "CF-Seg",
        "Matched",
        "Reverse",
        "Non-matched",
        "Boundary",
        "Context",
    ]
    header_height = 30
    label_height = 24
    legend_height = 30
    tile_width = args.tile_width
    font = load_font(12)
    title_font = load_font(14)

    sample = original.loc[case_ids[0]]
    sample_tile = overlay(
        resolve(sample["image_path"], project_root),
        resolve(sample["lung_mask_path"], project_root),
        resolve(sample["medsam_original_mask_path"], project_root),
        parse_box(sample["box_xyxy"]),
        tile_width,
    )
    tile_height = sample_tile.height + label_height
    canvas = Image.new("RGB", (tile_width * len(columns), header_height + tile_height * len(case_ids) + legend_height), "white")
    draw = ImageDraw.Draw(canvas)

    for col_idx, label in enumerate(columns):
        x0 = col_idx * tile_width
        draw.rectangle((x0, 0, x0 + tile_width, header_height), fill=(245, 245, 242))
        draw.text((x0 + 5, 7), label, fill=(20, 20, 20), font=title_font)

    for row_idx, case_id in enumerate(case_ids):
        original_row = original.loc[case_id]
        cfseg_row = cfseg.loc[case_id]
        box = parse_box(original_row["box_xyxy"])
        reference = resolve(original_row["lung_mask_path"], project_root)
        original_image = resolve(original_row["image_path"], project_root)
        cfseg_image = resolve(cfseg_row["counterfactual_path"], project_root)
        specs = [
            (original_image, original_row["medsam_original_mask_path"]),
            (cfseg_image, cfseg_row["medsam_counterfactual_mask_path"]),
            (original_image, first_patch(patching, case_id, "matched_cf_to_original", args.layer, "whole")),
            (cfseg_image, first_patch(patching, case_id, "reverse_original_to_cf", args.layer, "whole")),
            (original_image, first_patch(patching, case_id, "nonmatched_cf_to_original", args.layer, "whole")),
            (original_image, first_patch(patching, case_id, "matched_cf_to_original", args.layer, "boundary")),
            (original_image, first_patch(patching, case_id, "matched_cf_to_original", args.layer, "background")),
        ]
        y0 = header_height + row_idx * tile_height
        for col_idx, (image_path, prediction_path) in enumerate(specs):
            tile = overlay(resolve(image_path, project_root), reference, resolve(prediction_path, project_root), box, tile_width)
            x0 = col_idx * tile_width
            canvas.paste(tile, (x0, y0))
            draw.text((x0 + 5, y0 + tile.height + 5), case_id.replace("nih_", ""), fill=(35, 35, 35), font=font)

    legend_y = header_height + tile_height * len(case_ids) + 7
    draw.text(
        (8, legend_y),
        "Orange: CheXMask weak/silver boundary. Cyan: MedSAM prediction. Yellow: fixed prompt box.",
        fill=(35, 35, 35),
        font=font,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output)
    print(f"Wrote {args.output} for cases: {', '.join(case_ids)}")


if __name__ == "__main__":
    main()
