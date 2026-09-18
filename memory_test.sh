#!/usr/bin/env bash
# ProactiveClaw — wrapper for src/memory_monitor/test_writer.py
# Usage: ./memory_test.sh [--delay N] [--batch]
set -euo pipefail
# shellcheck disable=SC1091
source "$(dirname "${BASH_SOURCE[0]}")/bash_scripts/_venv.sh"
exec python "$ROOT/src/memory_monitor/test_writer.py" "$@"
