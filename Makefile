HOST  := 127.0.0.1
PORT  := 9099
URL   := http://$(HOST):$(PORT)
PID_FILE := .proxy.pid

# Optional: export LANGSMITH_API_KEY=ls__... in your shell to enable tracing.
# Get a free key at https://smith.langchain.com → Settings → API Keys
LANGSMITH_PROJECT ?= llm-compressor

.PHONY: install install-langsmith start restart stop dashboard stats rtk-stats check langsmith-status langsmith-test help

help:
	@echo "Usage: make <target>"
	@echo ""
	@echo "  install            Install core dependencies via uv"
	@echo "  install-langsmith  Install core + langsmith optional dep"
	@echo "  start              Start the proxy (requires ANTHROPIC_API_KEY)"
	@echo "  stop               Stop the running proxy"
	@echo "  restart            Stop and start fresh"
	@echo "  dashboard          Open the dashboard in the browser"
	@echo "  stats              Print proxy compression stats (JSON)"
	@echo "  rtk-stats          Print rtk shell-layer savings"
	@echo "  check              Check proxy is reachable"
	@echo "  langsmith-status   Check if LangSmith tracing is active on running proxy"
	@echo "  langsmith-test     Send a test request and check LangSmith received it"
	@echo ""
	@echo "LangSmith (optional): export LANGSMITH_API_KEY=ls__... then make start"

install:
	uv sync

install-langsmith:
	uv sync --group langsmith

start:
	@if [ -z "$$ANTHROPIC_API_KEY" ]; then \
		echo "Error: ANTHROPIC_API_KEY is not set"; exit 1; \
	fi
	LANGSMITH_PROJECT=$(LANGSMITH_PROJECT) uv run python proxy.py

stop:
	@if [ -f $(PID_FILE) ]; then \
		PID=$$(cat $(PID_FILE)); \
		kill $$PID 2>/dev/null && echo "Stopped PID $$PID" || echo "PID $$PID not running"; \
		rm -f $(PID_FILE); \
	else \
		echo "No $(PID_FILE) found — nothing to stop"; \
	fi
	@lsof -ti tcp:$(PORT) -sTCP:LISTEN | xargs kill -9 2>/dev/null || true

restart: stop
	@if [ -z "$$ANTHROPIC_API_KEY" ]; then \
		echo "Error: ANTHROPIC_API_KEY is not set"; exit 1; \
	fi
	@nohup LANGSMITH_PROJECT=$(LANGSMITH_PROJECT) uv run python proxy.py >> proxy.log 2>&1 & echo $$! > $(PID_FILE) && echo "Started PID $$(cat $(PID_FILE))"

dashboard:
	open $(URL)/dashboard

stats:
	@curl -s $(URL)/stats | python3 -m json.tool

rtk-stats:
	rtk gain

check:
	@curl -sf $(URL)/ > /dev/null && echo "Proxy is up at $(URL)" || echo "Proxy is not running"

langsmith-status:
	@curl -s $(URL)/admin/langsmith-status | python3 -m json.tool

langsmith-test:
	@if [ -z "$$ANTHROPIC_API_KEY" ]; then \
		echo "Error: ANTHROPIC_API_KEY is not set"; exit 1; \
	fi
	@echo "Sending test request through proxy..."
	@curl -s -X POST $(URL)/v1/messages \
		-H "Content-Type: application/json" \
		-H "x-api-key: $$ANTHROPIC_API_KEY" \
		-H "anthropic-version: 2023-06-01" \
		-d '{"model":"claude-haiku-4-5","max_tokens":30,"messages":[{"role":"user","content":"Say hi in one word."}]}' \
		| python3 -m json.tool
	@echo ""
	@echo "Check LangSmith: https://smith.langchain.com/o/projects (project: $(LANGSMITH_PROJECT))"
