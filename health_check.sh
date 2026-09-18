#!/usr/bin/env bash
# ProactiveClaw — wrapper for src/health_check/check.py
# Usage: ./health_check.sh [--fast] [--json]
set -euo pipefail
# shellcheck disable=SC1091
source "$(dirname "${BASH_SOURCE[0]}")/bash_scripts/_venv.sh"
exec python "$ROOT/src/health_check/check.py" "$@"
