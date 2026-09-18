#!/usr/bin/env bash
# Clear all long-term memories from the Mem0/Qdrant store.
#
# Usage:
#   ./bash_scripts/reset/clear_memory.sh [--force] [--wipe]
#
#   --force   Skip confirmation prompt
#   --wipe    Delete mem0_data files directly instead of using the Mem0 API

set -euo pipefail

# shellcheck disable=SC1091
source "$(dirname "${BASH_SOURCE[0]}")/../_venv.sh"
SCRIPT="$ROOT/src/reset/clear_memory.py"
exec python "$SCRIPT" "$@"
