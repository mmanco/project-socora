#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="${1:-output}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${SCRIPT_DIR}/.."

if [[ -f "${ROOT_DIR}/pyproject.toml" ]]; then
  cd "${ROOT_DIR}"
fi

if [[ ! -d "${BASE_DIR}" ]]; then
  echo "[ERROR] Base directory not found: ${BASE_DIR}" >&2
  exit 2
fi

# Use existing environment; do not resync per invocation
export UV_NO_SYNC="${UV_NO_SYNC:-1}"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"

uv run --no-sync python scripts/find_empty_content_md.py "${BASE_DIR}"

