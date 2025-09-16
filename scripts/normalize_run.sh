#!/usr/bin/env bash
set -euo pipefail

# cd to repo root (dir containing pyproject.toml)
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${SCRIPT_DIR}/.."
if [[ -f "${ROOT_DIR}/pyproject.toml" ]]; then
  cd "${ROOT_DIR}"
else
  if [[ -f pyproject.toml ]]; then
    : # already in root
  else
    ROOT_FROM_GIT=$(git rev-parse --show-toplevel 2>/dev/null || true)
    if [[ -n "${ROOT_FROM_GIT}" && -f "${ROOT_FROM_GIT}/pyproject.toml" ]]; then
      cd "${ROOT_FROM_GIT}"
    fi
  fi
fi

# First arg can be a run dir; otherwise detect latest under output/run-*
RUN_DIR=""
if [[ $# -gt 0 && "$1" != -* ]]; then
  RUN_DIR="$1"
  shift
fi

if [[ -z "${RUN_DIR}" ]]; then
  if [[ -d output ]]; then
    # shellcheck disable=SC2012
    RUN_DIR=$(ls -1dt output/run-* 2>/dev/null | head -n1 || true)
  fi
fi

if [[ -z "${RUN_DIR}" || ! -d "${RUN_DIR}" ]]; then
  echo "Usage: scripts/normalize_run.sh [output/run-YYYYmmdd-HHMMSS] [--force-commonalities] [--common-threshold 0.5]" >&2
  exit 1
fi

echo "Normalizing run: ${RUN_DIR}"

export PYTHONIOENCODING=utf-8

# Process each page directory once, preferring text_content.json over content.txt
while IFS= read -r -d '' d; do
  if [[ -f "$d/text_content.json" ]]; then
    echo "- $d/text_content.json"
    uv run python -m socora_crawler.normalize_text_content "$d/text_content.json" "$@" > "$d/content.md"
  elif [[ -f "$d/content.txt" ]]; then
    echo "- $d/content.txt"
    uv run python -m socora_crawler.normalize_text_content "$d/content.txt" "$@" > "$d/content.md"
  else
    echo "[WARN] Missing text_content.json and content.txt: $d"
  fi
done < <(find "${RUN_DIR}" -mindepth 1 -maxdepth 1 -type d -print0)

echo "Done."
