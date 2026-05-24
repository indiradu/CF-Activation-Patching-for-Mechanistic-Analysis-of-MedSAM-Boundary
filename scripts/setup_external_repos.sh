#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-external}"
mkdir -p "${ROOT}"

clone_if_missing() {
  local url="$1"
  local dest="$2"

  if [ -d "${dest}/.git" ]; then
    echo "Already exists: ${dest}"
    return
  fi

  echo "Cloning ${url} -> ${dest}"
  git clone "${url}" "${dest}"
}

clone_if_missing "https://github.com/bowang-lab/MedSAM.git" "${ROOT}/MedSAM"
clone_if_missing "https://github.com/biomedia-mira/CF-Seg.git" "${ROOT}/CF-Seg"

# Optional later baselines/generators. Keeping them here documents the intended stack.
clone_if_missing "https://github.com/OpenGVLab/SAM-Med2D.git" "${ROOT}/SAM-Med2D"
clone_if_missing "https://github.com/Amarkr1/PRISM.git" "${ROOT}/PRISM"

echo "External repositories are ready under ${ROOT}"

