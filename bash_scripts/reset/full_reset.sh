#!/usr/bin/env bash
# Full system reset — wipes everything back to a clean slate.
#
# What gets cleared:
#   • All long-term memories (Mem0/Qdrant)
#   • Invocation queue, reminders, re-engagement tracker
#   • Cron jobs (protected jobs kept unless --nuke)
#   • Session history files
#   • Agent file system contents (user_preferences.json kept unless --nuke)
#   • Subagent registry
#   • Daily brief and last session summary
#
# Usage:
#   ./bash_scripts/reset/full_reset.sh [--force] [--nuke] [--dry-run]
#
#   --force    Skip all confirmation prompts
#   --nuke     Also wipe protected cron jobs and user_preferences.json
#   --dry-run  Show what would be deleted without actually doing anything

set -euo pipefail

# shellcheck disable=SC1091
source "$(dirname "${BASH_SOURCE[0]}")/../_venv.sh"
SCRIPT="$ROOT/src/reset/full_reset.py"
exec python "$SCRIPT" "$@"
