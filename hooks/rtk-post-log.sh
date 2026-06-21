#!/usr/bin/env bash
# PostToolUse hook: log RTK savings to llmlingua proxy with session attribution.
#
# Flow:
#   1. Claude Code ran "rtk <cmd> ..." (rewritten by the PreToolUse rtk hook)
#   2. RTK wrote exactly one row to history.db before this hook fires
#   3. We lock, query the most recent matching row (within 5s), POST to /rtk/log
#
# Install: add to ~/.claude/settings.json under hooks.PostToolUse for Bash matcher.

set -euo pipefail

PROXY_URL="http://127.0.0.1:9099/rtk/log"

case "$(uname -s)" in
    Darwin) RTK_DB="$HOME/Library/Application Support/rtk/history.db" ;;
    *)      RTK_DB="$HOME/.local/share/rtk/history.db" ;;
esac

INPUT=$(cat)

# Debug: log received payload to /tmp for inspection
printf '%s' "$INPUT" > /tmp/rtk_post_debug.json

# PostToolUse receives the REWRITTEN command (after rtk PreToolUse hook rewrote it)
# e.g. "rtk git status" not "git status"
CMD=$(printf '%s' "$INPUT" | jq -r '.tool_input.command // ""')
SESSION_ID=$(printf '%s' "$INPUT" | jq -r '.session_id // "unknown"')

[[ -z "$CMD" || ! -f "$RTK_DB" ]] && exit 0

# Serialize access via atomic mkdir lock (portable on macOS + Linux)
# trap ensures the lock is released even if the script exits unexpectedly
LOCK_DIR="/tmp/rtk_proxy_log.lock.d"
trap 'rmdir "$LOCK_DIR" 2>/dev/null || true' EXIT
while ! mkdir "$LOCK_DIR" 2>/dev/null; do sleep 0.02; done

# Match by rtk_cmd (the rewritten command Claude Code actually executed)
ESCAPED="${CMD/\'/\'\'}"
ROW=$(sqlite3 "$RTK_DB" \
    "SELECT id, rtk_cmd, input_tokens, output_tokens, saved_tokens, savings_pct, timestamp
     FROM commands
     WHERE rtk_cmd = '${ESCAPED}'
       AND timestamp >= strftime('%Y-%m-%dT%H:%M:%S', datetime('now', '-5 seconds'))
     ORDER BY id DESC
     LIMIT 1")

[[ -z "${ROW:-}" ]] && exit 0

IFS='|' read -r ROW_ID ROW_CMD INP OUT SAVED PCT TS <<< "$ROW"

curl -sf --max-time 2 -X POST "$PROXY_URL" \
    -H "Content-Type: application/json" \
    -d "{\"rtk_id\":${ROW_ID},\"ts\":\"${TS}\",\"session_id\":\"${SESSION_ID}\",\"rtk_cmd\":\"${ROW_CMD}\",\"input_tokens\":${INP},\"output_tokens\":${OUT},\"saved_tokens\":${SAVED},\"savings_pct\":${PCT}}" \
    > /dev/null 2>&1 || true

exit 0
