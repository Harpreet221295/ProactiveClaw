#!/usr/bin/env bash
# Memory Conversation Simulator — test graph-backed memory without running the full agent.
#
# Two modes:
#
#   Interactive (default):
#     You type user messages, optionally type the agent response, and each
#     turn is stored via store_dialogues() — exactly how the real agent does it.
#     After each store, shows what memory items were extracted and what new
#     graph edges appeared.
#
#   Scripted (--script):
#     Feeds pre-written conversation pairs automatically with a delay,
#     so you can focus on watching the graph update on another terminal.
#
# Terminal layout:
#   Terminal 1:  ./bash_scripts/memory_simulator.sh
#   Terminal 2:  ./memory_monitor.sh --watch-graph
#
# Usage:
#   ./bash_scripts/memory_simulator.sh                  # interactive
#   ./bash_scripts/memory_simulator.sh --script         # scripted pairs
#   ./bash_scripts/memory_simulator.sh --script --delay 6

set -euo pipefail

# shellcheck disable=SC1091
source "$(dirname "${BASH_SOURCE[0]}")/_venv.sh"
SCRIPT="$ROOT/src/memory_monitor/convo_simulator.py"
exec python "$SCRIPT" "$@"
