#!/usr/bin/env bash
# Clear notification queues, reminders, and re-engagement tracker.
#
# Usage:
#   ./bash_scripts/reset/clear_queues.sh [--force] [--queue] [--reminders] [--reengagement]
#
#   No flags = clear all three.
#   --force         Skip confirmation prompts
#   --queue         Clear invocation queue only
#   --reminders     Clear reminders only
#   --reengagement  Clear re-engagement tracker only

set -euo pipefail

# shellcheck disable=SC1091
source "$(dirname "${BASH_SOURCE[0]}")/../_venv.sh"
SCRIPT="$ROOT/src/reset/clear_queues.py"
exec python "$SCRIPT" "$@"
