#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

uv run python run_all_sites.py --rescan-only >> "$ROOT/watcher.log" 2>&1
