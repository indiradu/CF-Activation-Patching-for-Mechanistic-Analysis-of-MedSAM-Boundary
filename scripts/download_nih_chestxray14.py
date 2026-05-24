#!/usr/bin/env python3
"""Download NIH ChestX-ray14 from Kaggle using KaggleHub.

Run this on the GPU/scratch machine, not on a slow local connection.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--handle", default="nih-chest-xrays/data")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["KAGGLEHUB_CACHE"] = str(args.cache_dir.resolve())

    import kagglehub

    path = kagglehub.dataset_download(args.handle)
    print(f"Path to dataset files: {path}")


if __name__ == "__main__":
    main()

