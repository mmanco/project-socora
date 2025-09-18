#!/usr/bin/env bash
set -euo pipefail

# cd to project root
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${SCRIPT_DIR}/.."
if [[ -f "${ROOT_DIR}/pyproject.toml" ]]; then
  cd "${ROOT_DIR}"
else
  if [[ -f pyproject.toml ]]; then :; else
    ROOT_FROM_GIT=$(git rev-parse --show-toplevel 2>/dev/null || true)
    if [[ -n "${ROOT_FROM_GIT}" && -f "${ROOT_FROM_GIT}/pyproject.toml" ]]; then
      cd "${ROOT_FROM_GIT}"
    fi
  fi
fi

RUN_DIR=""
if [[ $# -gt 0 && "$1" != -* ]]; then
  RUN_DIR="$1"; shift
fi
if [[ -z "${RUN_DIR}" && -d output ]]; then
  RUN_DIR=$(ls -1dt output/run-* 2>/dev/null | head -n1 || true)
fi
if [[ -z "${RUN_DIR}" || ! -d "${RUN_DIR}" ]]; then
  echo "Usage: scripts/extract_links_run.sh [output/run-YYYYmmdd-HHMMSS]" >&2
  exit 1
fi

echo "Extracting links for run: ${RUN_DIR}"
# Iterate page directories; prefer text_content.json else content.txt; else skip
find "${RUN_DIR}" -mindepth 1 -maxdepth 1 -type d -print0 | while IFS= read -r -d '' d; do
  if [[ -f "$d/content.json" ]]; then
    echo "- $d/content.json"
    uv run python -m socora_crawler.extract_links "$d/content.json" --write >/dev/null
  elif [[ -f "$d/content.txt" ]]; then
    echo "- $d/content.txt"
    uv run python -m socora_crawler.extract_links "$d/content.txt" --write >/dev/null
  else
    echo "- $d (no content files; using metadata)"
    uv run python -m socora_crawler.extract_links "$d" --write >/dev/null
  fi
done

# Aggregate JSONL
uv run python -m socora_crawler.aggregate_links "${RUN_DIR}"

echo "Done."
