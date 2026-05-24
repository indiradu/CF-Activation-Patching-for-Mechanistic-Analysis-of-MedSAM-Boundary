#!/usr/bin/env python3
"""Recover activation-patching metrics from saved patch-mask PNGs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from counterfactual_medsam.metrics import evaluate_mask_pair


DEFAULT_REMOTE_PROJECT_PREFIX = "/path/to/project"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--patch-mask-dir", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--case-column", default="case_id")
    parser.add_argument("--reference-column", default="lung_mask_path")
    parser.add_argument(
        "--remote-project-prefix",
        default=DEFAULT_REMOTE_PROJECT_PREFIX,
        help="Optional prefix to strip from absolute paths when recovering cluster outputs.",
    )
    return parser.parse_args()


def resolve_path(value: str, project_root: Path, remote_project_prefix: str) -> Path:
    path = Path(str(value))
    if path.exists():
        return path
    text = str(value)
    if remote_project_prefix and text.startswith(remote_project_prefix):
        candidate = project_root / text.removeprefix(remote_project_prefix).lstrip("/")
        if candidate.exists():
            return candidate
    candidate = project_root / text
    if candidate.exists():
        return candidate
    raise FileNotFoundError(value)


def parse_patch_name(path: Path) -> tuple[str, str, str, str]:
    parts = path.stem.split("__")
    if len(parts) != 4:
        raise ValueError(f"Unexpected patch-mask filename: {path.name}")
    case_id, control, layer, region = parts
    return case_id, control, layer, region


def main() -> None:
    args = parse_args()
    project_root = Path.cwd()
    manifest = pd.read_csv(args.manifest)
    references = {
        str(row[args.case_column]): resolve_path(
            str(row[args.reference_column]),
            project_root,
            args.remote_project_prefix,
        )
        for _, row in manifest.iterrows()
    }

    rows = []
    for patch_path in sorted(args.patch_mask_dir.glob("*.png")):
        case_id, control, layer, region = parse_patch_name(patch_path)
        if case_id not in references:
            continue
        metrics = evaluate_mask_pair(
            prediction_path=patch_path,
            reference_path=references[case_id],
        )
        rows.append(
            {
                "case_id": case_id,
                "control": control,
                "target_image": "counterfactual" if control == "reverse_original_to_cf" else "original",
                "donor_image": {
                    "matched_cf_to_original": "matched_counterfactual",
                    "reverse_original_to_cf": "matched_original",
                    "nonmatched_cf_to_original": "nonmatched_counterfactual",
                }.get(control, ""),
                "donor_case_id": "",
                "layer": layer,
                "region": region,
                "prediction_path": str(patch_path),
                **metrics,
            }
        )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output_csv, index=False)
    print(f"Wrote {len(rows)} recovered rows to {args.output_csv}")


if __name__ == "__main__":
    main()
