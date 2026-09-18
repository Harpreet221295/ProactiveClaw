#!/usr/bin/env bash
# Start ProactiveClaw.
#   ./run.sh          web UI + proactive server at http://127.0.0.1:8000
#   ./run.sh --cli    terminal chat only (no nudges delivered)
#   ./run.sh --check  run the health check first
set -euo pipefail
# shellcheck disable=SC1091
source "$(dirname "${BASH_SOURCE[0]}")/bash_scripts/_venv.sh"
cd "$ROOT"
if [[ ! -f .env ]]; then
    echo "No .env found. Run ./setup.sh first."; exit 1
fi
case "${1:-}" in
    --cli)   exec python src/main.py ;;
    --check) python src/health_check/check.py --fast || true; exec python src/serve.py ;;
    *)       exec python src/serve.py ;;
esac
