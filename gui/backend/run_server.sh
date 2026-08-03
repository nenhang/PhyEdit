#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON_BIN="${GUI_PYTHON:-python}"

cd "$PROJECT_ROOT"
exec "$PYTHON_BIN" -m uvicorn gui.backend.app:app \
  --host "${GUI_HOST:-0.0.0.0}" \
  --port "${GUI_PORT:-8000}" \
  "$@"
