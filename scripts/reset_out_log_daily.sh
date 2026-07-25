#!/usr/bin/env bash
# Сброс out.log всех сайтов к пустому файлу с заголовком на новые сутки.
#
# Cron (00:00):
#   0 0 * * * /var/www/app/log-analyzer-multi-site/scripts/reset_out_log_daily.sh >> /var/www/app/log-analyzer-multi-site/watcher.log 2>&1

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -f "$ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT/.env"
  set +a
fi

if [[ "$#" -gt 0 ]]; then
  exec uv run python "$ROOT/scripts/reset_out_log_daily.py" "$@"
fi

exec uv run python "$ROOT/scripts/reset_out_log_daily.py" --all-sites
