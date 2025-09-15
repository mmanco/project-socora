#!/usr/bin/env bash
set -euo pipefail

# Stop and remove the Tika container via docker-compose
cd "$(dirname "$0")"
docker-compose down

echo "Tika has been stopped."
