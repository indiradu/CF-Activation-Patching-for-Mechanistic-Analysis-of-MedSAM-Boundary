#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${1:-/scratch/$USER/miccai_workshop/chexmask}"
VERSION="${CHEXMASK_VERSION:-1.0.0}"
BASE_URL="https://physionet.org/files/chexmask-cxr-segmentation-data/${VERSION}"
ROOT="${OUTPUT_DIR}/chexmask-cxr-segmentation-data-${VERSION}"

mkdir -p "${ROOT}/OriginalResolution"

download_one() {
  local rel_path="$1"
  local url="${BASE_URL}/${rel_path}"
  local dest="${ROOT}/${rel_path}"
  mkdir -p "$(dirname "${dest}")"
  echo "Downloading ${url}"
  wget -c --progress=bar:force:noscroll -O "${dest}" "${url}"
}

# NIH ChestX-ray14 corresponds to ChestX-Ray8 in CheXMask.
download_one "OriginalResolution/ChestX-Ray8.csv"
download_one "DATA_DICTIONARY.md"
download_one "README.md"
download_one "LICENSE.txt"
download_one "SHA256SUMS.txt"

echo "CheXMask NIH subset ready at ${ROOT}"
