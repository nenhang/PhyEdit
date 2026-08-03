#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DA3_DIR="${1:-${REPO_ROOT}/third_party/Depth-Anything-3}"
DA3_COMMIT="2c21ea849ceec7b469a3e62ea0c0e270afc3281a"
PATCH_PATH="${REPO_ROOT}/patches/depth_anything_3_phyedit.patch"
PYTHON_BIN="${PYTHON:-python}"

if [[ ! -d "${DA3_DIR}/.git" ]]; then
  mkdir -p "$(dirname "${DA3_DIR}")"
  git clone https://github.com/ByteDance-Seed/Depth-Anything-3.git "${DA3_DIR}"
else
  if ! git -C "${DA3_DIR}" diff --quiet || ! git -C "${DA3_DIR}" diff --cached --quiet; then
    echo "Refusing to modify a dirty Depth-Anything-3 checkout: ${DA3_DIR}" >&2
    exit 1
  fi
  git -C "${DA3_DIR}" fetch origin
fi

git -C "${DA3_DIR}" checkout --detach "${DA3_COMMIT}"

if git -C "${DA3_DIR}" apply --reverse --check "${PATCH_PATH}" >/dev/null 2>&1; then
  echo "PhyEdit Depth Anything 3 patch is already applied."
else
  git -C "${DA3_DIR}" apply --check "${PATCH_PATH}"
  git -C "${DA3_DIR}" apply "${PATCH_PATH}"
fi

"${PYTHON_BIN}" -m pip install -e "${DA3_DIR}"
echo "Installed the PhyEdit-compatible Depth Anything 3 checkout at ${DA3_DIR}"
