#!/usr/bin/env bash
# Clear cron jobs. Protected jobs (cron_deep_think) are preserved by default.
#
# Usage:
#   ./bash_scripts/reset/clear_cron_jobs.sh [--force] [--all] [--list]
#
#   --force  Skip confirmation prompt
#   --all    Also delete protected jobs (e.g. cron_deep_think)
#   --list   Just list current jobs without deleting

set -euo pipefail

# shellcheck disable=SC1091
source "$(dirname "${BASH_SOURCE[0]}")/../_venv.sh"
SCRIPT="$ROOT/src/reset/clear_cron_jobs.py"
exec python "$SCRIPT" "$@"
