#!/usr/bin/env python3
"""Validate the local path configuration without exposing private paths in Git."""

from __future__ import annotations

import argparse
from pathlib import Path

from counterfactual_medsam.config import flatten_mapping, load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    flat = flatten_mapping(cfg)

    missing_required = []
    for key, value in flat.items():
        if value is None or not isinstance(value, str):
            continue
        path = Path(value).expanduser()
        looks_like_path = value.startswith(("/", "~", ".")) or "/" in value
        if not looks_like_path:
            continue
        exists = path.exists()
        status = "OK" if exists else "MISSING"
        print(f"{status:7s} {key}: {path}")
        if key in {
            "data.mimic_cxr_jpg_root",
            "external_repos.medsam",
            "checkpoints.medsam_vit_b",
        } and not exists:
            missing_required.append(key)

    if missing_required:
        missing = ", ".join(missing_required)
        raise SystemExit(f"Missing required first-milestone paths: {missing}")


if __name__ == "__main__":
    main()

