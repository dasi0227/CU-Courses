#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="$PROJECT_DIR/.venv/bin/python"
cd "$PROJECT_DIR"

"$PYTHON_BIN" -m src.simulation --seller_messages_enabled --ceo_pressure --tag ceo_pressure --seeds {1..10}
"$PYTHON_BIN" -m src.simulation --seller_messages_enabled --overseer_enabled --tag oversight --seeds {1..10}
"$PYTHON_BIN" -m src.simulation --seller_messages_enabled --ceo_pressure --overseer_enabled \
    --tag ceo_pressure-oversight --seeds {1..10}
