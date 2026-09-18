#!/usr/bin/env bash
# Shared helper: locate and activate the project's virtualenv.
# Sourced by the other scripts. Sets $ROOT and activates the venv.
_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$_here/.." && pwd)"
VENV="${VENV_DIR:-}"
if [[ -z "$VENV" ]]; then
    for cand in "$ROOT/.venv" "$ROOT/proactive_claw_venv" "$ROOT/venv"; do
        if [[ -f "$cand/bin/activate" ]]; then VENV="$cand"; break; fi
    done
fi
if [[ -z "$VENV" || ! -f "$VENV/bin/activate" ]]; then
    echo "No virtualenv found. Run ./setup.sh first (it creates .venv and installs dependencies)."
    exit 1
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
export ROOT VENV
