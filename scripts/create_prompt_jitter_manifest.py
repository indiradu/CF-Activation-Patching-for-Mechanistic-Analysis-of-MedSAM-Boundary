#!/usr/bin/env python3
"""Create a manifest with deterministic jittered MedSAM box prompts."""

from __future__ import annotations

import argparse
import zlib
from pathlib import Path

import pandas as pd
from PIL import Image

from counterfactual_medsam.manifest import parse_box_xyxy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--pct", required=True, type=float, help="Jitter level, e.g. 0.05 for 5%.")
    parser.add_argument("--image-column", default="image_path")
    parser.add_argument("--box-column", default="box_xyxy")
    parser.add_argument("--case-column", default="case_id")
    parser.add_argument(
        "--mode",
        choices=["expand", "translate", "expand_translate"],
        default="expand_translate",
        help="Prompt perturbation type. The default combines expansion and translation.",
    )
    return parser.parse_args()


def stable_unit_interval(*parts: object) -> float:
    value = zlib.crc32("|".join(str(part) for part in parts).encode("utf-8")) & 0xFFFFFFFF
    return value / 0xFFFFFFFF


def resolve_path(path_value: object, manifest_path: Path) -> Path:
    path = Path(str(path_value))
    if path.is_absolute():
        return path
    return manifest_path.parent.parent.parent / path if str(path).startswith("data/") else path


def clamp_box(x0: float, y0: float, x1: float, y1: float, width: int, height: int) -> tuple[int, int, int, int]:
    x0 = max(0.0, min(float(width - 2), x0))
    y0 = max(0.0, min(float(height - 2), y0))
    x1 = max(x0 + 1.0, min(float(width - 1), x1))
    y1 = max(y0 + 1.0, min(float(height - 1), y1))
    return int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1))


def jitter_box(
    box: tuple[int, int, int, int],
    *,
    width: int,
    height: int,
    pct: float,
    case_id: str,
    mode: str,
) -> tuple[tuple[int, int, int, int], float, float, float, float]:
    x0, y0, x1, y1 = box
    box_w = max(1.0, float(x1 - x0))
    box_h = max(1.0, float(y1 - y0))

    expand_x = pct * box_w if mode in {"expand", "expand_translate"} else 0.0
    expand_y = pct * box_h if mode in {"expand", "expand_translate"} else 0.0
    dx = 0.0
    dy = 0.0
    if mode in {"translate", "expand_translate"}:
        dx_unit = stable_unit_interval(case_id, pct, "dx") * 2.0 - 1.0
        dy_unit = stable_unit_interval(case_id, pct, "dy") * 2.0 - 1.0
        dx = dx_unit * pct * box_w
        dy = dy_unit * pct * box_h

    jittered = clamp_box(
        x0 - expand_x + dx,
        y0 - expand_y + dy,
        x1 + expand_x + dx,
        y1 + expand_y + dy,
        width,
        height,
    )
    return jittered, dx, dy, expand_x, expand_y


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.manifest)
    rows = []
    for _, row in df.iterrows():
        case_id = str(row[args.case_column])
        image_path = resolve_path(row[args.image_column], args.manifest)
        with Image.open(image_path) as image:
            width, height = image.size
        original_box = parse_box_xyxy(row[args.box_column])
        jittered, dx, dy, expand_x, expand_y = jitter_box(
            original_box,
            width=width,
            height=height,
            pct=args.pct,
            case_id=case_id,
            mode=args.mode,
        )
        updated = row.to_dict()
        updated[f"{args.box_column}_original"] = row[args.box_column]
        updated[args.box_column] = ",".join(str(value) for value in jittered)
        updated["prompt_jitter_pct"] = args.pct
        updated["prompt_jitter_mode"] = args.mode
        updated["prompt_jitter_dx_px"] = dx
        updated["prompt_jitter_dy_px"] = dy
        updated["prompt_jitter_expand_x_px"] = expand_x
        updated["prompt_jitter_expand_y_px"] = expand_y
        rows.append(updated)

    out = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"Wrote {len(out)} rows with {args.pct:.0%} {args.mode} prompt jitter to {args.output}")


if __name__ == "__main__":
    main()
