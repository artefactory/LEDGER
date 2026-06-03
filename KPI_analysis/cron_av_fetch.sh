#!/usr/bin/env bash
# Daily Alpha Vantage gap-fill drain (cron entrypoint).
# Re-runs the primary EDGAR/yfinance pass (cached, cheap) and lets
# fetch_kpis.py spend the day's AV quota on the worst-coverage tickers.
#
# Designed to be invoked by `crontab -e` running as the user (no sudo).
# Single-instance via flock; appends to a date-stamped log under
# KPI_analysis/output/cron_logs/.

set -euo pipefail

REPO="/home/cmoslonka/ardian_dataset_bench"
LOG_DIR="${REPO}/KPI_analysis/output/cron_logs"
LOCK_FILE="${REPO}/KPI_analysis/cache/cron_av_fetch.lock"

mkdir -p "$LOG_DIR" "$(dirname "$LOCK_FILE")"

# cron(8) sets a minimal PATH; uv lives in the user's local bin.
export PATH="/home/cmoslonka/.local/bin:/usr/local/bin:/usr/bin:/bin"

LOG_FILE="${LOG_DIR}/$(date -u +%Y-%m-%d).log"

cd "$REPO"

# Redirect everything from here on into the day's log.
exec >>"$LOG_FILE" 2>&1
echo "===== $(date -u +%FT%TZ) cron_av_fetch start ====="

# Single-instance lock; second invocation exits 0 without running.
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
    echo "another instance is running; exiting."
    exit 0
fi

rc=0
uv run python -m KPI_analysis.kpi_fetch_and_build.fetch_kpis \
    --selected \
    --years 2017-2022 \
    --alphavantage \
    || rc=$?

echo "===== $(date -u +%FT%TZ) cron_av_fetch end (rc=$rc) ====="
exit "$rc"
