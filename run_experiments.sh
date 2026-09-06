#!/usr/bin/env bash
# Truthful ScholAR evaluation entry point. The default profile is offline and deterministic.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
if [[ ! -x "$PYTHON_BIN" ]]; then
    PYTHON_BIN="python3"
fi

exec "$PYTHON_BIN" "$ROOT_DIR/evaluation/run_evaluation_profiles.py" "$@"
