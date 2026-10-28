#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="$PROJECT_DIR/.venv/bin/python"
cd "$PROJECT_DIR"

"$PYTHON_BIN" -m src.simulation --tag base --seeds {1..10}
"$PYTHON_BIN" -m src.simulation --seller_messages_enabled --tag seller_comms --seeds {1..10}
