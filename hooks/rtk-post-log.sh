#!/usr/bin/env bash
# PostToolUse hook: log RTK savings to llmlingua proxy with session attribution.
#
# Flow:
#   1. Claude Code ran a Bash command (rewritten to "rtk <cmd>" by the PreToolUse hook)
#   2. RTK wrote a row to history.db before this hook fires
#   3. We grab the most recent row within the last 10s and POST it to /rtk/log
#
# Note: tool_input.command in PostToolUse is the ORIGINAL command (before PreToolUse
# rewrote it), so we cannot match by command name — RTK stores "rtk <cmd>" while
# we receive "<cmd>". Instead we match purely by timestamp recency, which is reliable
# since this hook fires immediately after RTK exits. INSERT OR IGNORE on rtk_id
# prevents double-logging if two commands land in the same window.

set -euo pipefail

PROXY_URL="http://127.0.0.1:9099/rtk/log"

case "$(uname -s)" in
    Darwin) RTK_DB="$HOME/Library/Application Support/rtk/history.db" ;;
    *)      RTK_DB="$HOME/.local/share/rtk/history.db" ;;
esac

INPUT=$(cat)

SESSION_ID=$(printf '%s' "$INPUT" | jq -r '.session_id // "unknown"')

[[ ! -f "$RTK_DB" ]] && exit 0

# Serialize access via atomic mkdir lock (portable on macOS + Linux)
LOCK_DIR="/tmp/rtk_proxy_log.lock.d"
trap 'rmdir "$LOCK_DIR" 2>/dev/null || true' EXIT
while ! mkdir "$LOCK_DIR" 2>/dev/null; do sleep 0.02; done

# Match by recency only — command-name matching is unreliable because RTK truncates
# some commands and tool_input.command never has the "rtk " prefix.
ROW=$(sqlite3 "$RTK_DB" \
    "SELECT id, rtk_cmd, input_tokens, output_tokens, saved_tokens, savings_pct, timestamp, COALESCE(project_path,'')
     FROM commands
     WHERE timestamp >= strftime('%Y-%m-%dT%H:%M:%S', datetime('now', '-10 seconds'))
     ORDER BY id DESC
     LIMIT 1")

[[ -z "${ROW:-}" ]] && exit 0

IFS='|' read -r ROW_ID ROW_CMD INP OUT SAVED PCT TS PROJECT_PATH <<< "$ROW"

# Escape project_path for JSON (replace backslashes and double-quotes)
PROJECT_PATH_JSON=$(printf '%s' "${PROJECT_PATH}" | sed 's/\\/\\\\/g; s/"/\\"/g')

curl -sf --max-time 2 -X POST "$PROXY_URL" \
    -H "Content-Type: application/json" \
    -d "{\"rtk_id\":${ROW_ID},\"ts\":\"${TS}\",\"session_id\":\"${SESSION_ID}\",\"rtk_cmd\":\"${ROW_CMD}\",\"input_tokens\":${INP},\"output_tokens\":${OUT},\"saved_tokens\":${SAVED},\"savings_pct\":${PCT},\"project_path\":\"${PROJECT_PATH_JSON}\"}" \
    > /dev/null 2>&1 || true

exit 0
