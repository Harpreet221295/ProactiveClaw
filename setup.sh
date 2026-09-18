#!/usr/bin/env bash
# One-shot setup: create a virtualenv, install dependencies, run the setup wizard.
#
#   ./setup.sh            interactive (recommended for first run)
#   ./setup.sh --skip-wizard   just create the venv + install deps
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PY="${PYTHON:-python3}"
if ! command -v "$PY" >/dev/null 2>&1; then
    echo "python3 not found. Install Python 3.11+ and re-run."; exit 1
fi
ver="$("$PY" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
if [[ "$(printf '%s\n' "3.11" "$ver" | sort -V | head -n1)" != "3.11" ]]; then
    echo "Python 3.11+ required (found $ver)."; exit 1
fi

VENV="${VENV_DIR:-$ROOT/.venv}"
if [[ ! -f "$VENV/bin/activate" ]]; then
    echo "→ Creating virtualenv at $VENV"
    "$PY" -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
echo "→ Installing dependencies"
pip install --upgrade pip >/dev/null
pip install -r requirements.txt

mkdir -p config engagement_data agent_file_system sessions data
[[ -f .env ]] || cp .env.example .env
[[ -f config/config.json ]] || cp config/config.example.json config/config.json

if [[ "${1:-}" == "--skip-wizard" ]]; then
    echo "✓ Environment ready. Edit .env and config/config.json, then ./run.sh"
    exit 0
fi
python src/setup_wizard.py
