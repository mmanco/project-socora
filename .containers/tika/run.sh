#!/usr/bin/env bash
set -euo pipefail

# Run Apache Tika server in detached mode via docker-compose
cd "$(dirname "$0")"
docker-compose up -d

echo "Tika is starting on http://localhost:9998"
