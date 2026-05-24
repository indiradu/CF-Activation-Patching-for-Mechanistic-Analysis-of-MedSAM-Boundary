#!/usr/bin/env python3
"""Download CF-Seg release bundles from Google Drive."""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path


FILES = {
    "causal_gen_checkpoints": "13lCVYRnQco7G10t3Rwln6R1fLoRGIvQf",
    "chexmask_unet_checkpoints": "1pHkWmLfz8cOLLjO0-tNoFY6egKeff21g",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--only", choices=sorted(FILES), nargs="*")
    parser.add_argument("--unzip", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        import gdown
    except ImportError as exc:
        raise SystemExit("Install gdown first: python -m pip install gdown") from exc

    args.output_root.mkdir(parents=True, exist_ok=True)
    selected = args.only or sorted(FILES)
    for name in selected:
        file_id = FILES[name]
        output = args.output_root / f"{name}.download"
        print(f"Downloading {name} -> {output}")
        gdown.download(id=file_id, output=str(output), quiet=False)

        if args.unzip and zipfile.is_zipfile(output):
            extract_dir = args.output_root / name
            extract_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(output) as archive:
                archive.extractall(extract_dir)
            print(f"Extracted {output} -> {extract_dir}")


if __name__ == "__main__":
    main()
