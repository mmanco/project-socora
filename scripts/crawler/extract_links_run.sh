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
if [[ $# -gt 0 ]]; then
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
  echo "Usage: scripts/crawler/extract_links_run.sh [output/run-YYYYmmdd-HHMMSS]" >&2
  exit 1
fi

echo "Extracting links for run: ${RUN_DIR}"
export UV_NO_SYNC=1
export UV_LINK_MODE=copy

find "${RUN_DIR}" -mindepth 1 -maxdepth 1 -type d -print0 | while IFS= read -r -d '' d; do
  if [[ -f "$d/content.json" ]]; then
    echo "- $d/content.json"
    uv run --no-sync python -m socora_crawler.extract_links "$d/content.json" --write >/dev/null
  elif [[ -f "$d/content.txt" ]]; then
    echo "- $d/content.txt"
    uv run --no-sync python -m socora_crawler.extract_links "$d/content.txt" --write >/dev/null
  else
    echo "- $d (no content files; using metadata)"
    uv run --no-sync python -m socora_crawler.extract_links "$d" --write >/dev/null
  fi
done

uv run --no-sync python -m socora_crawler.aggregate_links "${RUN_DIR}"

echo "Done."