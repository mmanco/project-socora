#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
CRAWLER_DIR="${REPO_ROOT}/socora-crawler"
RUN_ROOT="${REPO_ROOT}/output"

if [[ ! -f "${CRAWLER_DIR}/pyproject.toml" ]]; then
  echo "[ERROR] Could not locate socora-crawler project at ${CRAWLER_DIR}" >&2
  exit 1
fi

cd "${CRAWLER_DIR}"

RUN_DIR=""
if [[ $# -gt 0 && $1 != -* ]]; then
  RUN_DIR="$1"
  shift
fi

if [[ -n "${RUN_DIR}" && "${RUN_DIR}" != /* ]]; then
  if [[ -d "${REPO_ROOT}/${RUN_DIR}" ]]; then
    RUN_DIR="${REPO_ROOT}/${RUN_DIR}"
  fi
fi

if [[ -z "${RUN_DIR}" ]]; then
  if [[ -d "${RUN_ROOT}" ]]; then
    RUN_DIR=$(ls -1dt "${RUN_ROOT}"/run-* 2>/dev/null | head -n1 || true)
  fi
fi

if [[ -z "${RUN_DIR}" || ! -d "${RUN_DIR}" ]]; then
  echo "Usage: scripts/crawler/normalize_run.sh [output/run-YYYYmmdd-HHMMSS] [--force-commonalities] [--common-threshold 0.5]" >&2
  exit 1
fi

echo "Normalizing run: ${RUN_DIR}"

export PYTHONIOENCODING=utf-8
export UV_NO_SYNC=1
export UV_LINK_MODE=copy

while IFS= read -r -d '' d; do
  if [[ -f "$d/content.json" ]]; then
    echo "- $d/content.json"
    uv run --no-sync python -m socora_crawler.normalize_content "$d/content.json" "$@" > "$d/content.md"
  elif [[ -f "$d/content.txt" ]]; then
    echo "- $d/content.txt"
    uv run --no-sync python -m socora_crawler.normalize_content "$d/content.txt" "$@" > "$d/content.md"
  else
    echo "[WARN] Missing content.json and content.txt: $d"
  fi
done < <(find "${RUN_DIR}" -mindepth 1 -maxdepth 1 -type d -print0)

echo "Done."