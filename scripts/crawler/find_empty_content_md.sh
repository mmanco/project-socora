#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
CRAWLER_DIR="${REPO_ROOT}/socora-crawler"
HELPER_PATH="${REPO_ROOT}/scripts/crawler/find_empty_content_md.py"

if [[ ! -f "${CRAWLER_DIR}/pyproject.toml" ]]; then
  echo "[ERROR] Could not locate socora-crawler project at ${CRAWLER_DIR}" >&2
  exit 1
fi

BASE_DIR="${1:-${REPO_ROOT}/output}"
shift 0

if [[ "${BASE_DIR}" != /* ]]; then
  if [[ -d "${REPO_ROOT}/${BASE_DIR}" ]]; then
    BASE_DIR="${REPO_ROOT}/${BASE_DIR}"
  fi
fi

if [[ ! -d "${BASE_DIR}" ]]; then
  echo "[ERROR] Base directory not found: ${BASE_DIR}" >&2
  exit 2
fi

cd "${CRAWLER_DIR}"

export UV_NO_SYNC=1
export UV_LINK_MODE=copy
export PYTHONIOENCODING=utf-8

uv run --no-sync python "${HELPER_PATH}" "${BASE_DIR}"