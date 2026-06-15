#!/usr/bin/env python3
"""Download the CheXMask subset needed for NIH ChestX-ray14 experiments.

The full CheXMask archive is large. For the NIH/ChestX-ray14 MVP we only need
the ChestX-Ray8 CSV, where CheXMask stores lung masks as RLE columns.
"""

from __future__ import annotations

import argparse
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


BASE_URL = "https://physionet.org/files/chexmask-cxr-segmentation-data/{version}"

SUBSET_FILES = {
    "nih": [
        "OriginalResolution/ChestX-Ray8.csv",
        "DATA_DICTIONARY.md",
        "README.md",
        "LICENSE.txt",
        "SHA256SUMS.txt",
    ],
    "all-original": [
        "OriginalResolution/ChestX-Ray8.csv",
        "OriginalResolution/CheXpert.csv",
        "OriginalResolution/MIMIC-CXR-JPG.csv",
        "OriginalResolution/Padchest.csv",
        "OriginalResolution/VinDr-CXR.csv",
        "DATA_DICTIONARY.md",
        "README.md",
        "LICENSE.txt",
        "SHA256SUMS.txt",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--version", default="1.0.0")
    parser.add_argument("--subset", choices=sorted(SUBSET_FILES), default="nih")
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--chunk-size", type=int, default=1024 * 1024)
    return parser.parse_args()


def remote_size(url: str) -> int | None:
    request = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            value = response.headers.get("Content-Length")
            return int(value) if value else None
    except urllib.error.HTTPError:
        return None


def fmt_bytes(value: int | None) -> str:
    if value is None:
        return "unknown"
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(value)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.1f}{unit}"
        size /= 1024
    return f"{value}B"


def download_one(url: str, destination: Path, *, retries: int, chunk_size: int) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    total = remote_size(url)
    existing = destination.stat().st_size if destination.exists() else 0

    if total is not None and existing == total:
        print(f"Already complete: {destination} ({fmt_bytes(total)})")
        return

    for attempt in range(1, retries + 1):
        try:
            headers = {}
            mode = "wb"
            if existing > 0:
                headers["Range"] = f"bytes={existing}-"
                mode = "ab"

            print(
                f"Downloading {url} -> {destination} "
                f"from {fmt_bytes(existing)} / {fmt_bytes(total)}"
            )
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=120) as response, destination.open(mode) as handle:
                if existing > 0 and response.status == 200:
                    print("Server did not honor resume request; restarting file.")
                    handle.close()
                    destination.unlink(missing_ok=True)
                    existing = 0
                    continue

                last_report = time.time()
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    handle.write(chunk)
                    existing += len(chunk)
                    now = time.time()
                    if now - last_report >= 10:
                        print(f"  progress: {fmt_bytes(existing)} / {fmt_bytes(total)}")
                        last_report = now

            if total is None or destination.stat().st_size == total:
                print(f"Finished: {destination} ({fmt_bytes(destination.stat().st_size)})")
                return

            existing = destination.stat().st_size
            print(f"Incomplete download, retrying: {fmt_bytes(existing)} / {fmt_bytes(total)}")
        except Exception as exc:  # noqa: BLE001 - command-line downloader should retry broad network errors.
            print(f"Attempt {attempt}/{retries} failed for {url}: {exc}", file=sys.stderr)
            time.sleep(min(60, 2**attempt))
            existing = destination.stat().st_size if destination.exists() else 0

    raise RuntimeError(f"Failed to download {url} after {retries} attempts")


def main() -> None:
    args = parse_args()
    base_url = BASE_URL.format(version=args.version).rstrip("/")
    for rel_path in SUBSET_FILES[args.subset]:
        url = f"{base_url}/{rel_path}"
        destination = args.output_dir / f"chexmask-cxr-segmentation-data-{args.version}" / rel_path
        download_one(url, destination, retries=args.retries, chunk_size=args.chunk_size)

    print("CheXMask subset download complete.")


if __name__ == "__main__":
    main()
