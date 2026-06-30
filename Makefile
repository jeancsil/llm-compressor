HOST  := 127.0.0.1
PORT  := 9099
URL   := http://$(HOST):$(PORT)
PID_FILE := .proxy.pid

# Optional: export LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY in your shell.
# Get keys at https://cloud.langfuse.com → Settings → API Keys
LANGFUSE_HOST ?= https://cloud.langfuse.com

.PHONY: install install-langfuse start restart stop dashboard stats rtk-stats check langfuse-status langfuse-test help

help:
	@echo "Usage: make <target>"
	@echo ""
	@echo "  install          Install core dependencies via uv"
	@echo "  install-langfuse Install core + langfuse optional dep"
	@echo "  start            Start the proxy (requires ANTHROPIC_API_KEY)"
	@echo "  stop             Stop the running proxy"
	@echo "  restart          Stop and start fresh"
	@echo "  dashboard        Open the dashboard in the browser"
	@echo "  stats            Print proxy compression stats (JSON)"
	@echo "  rtk-stats        Print rtk shell-layer savings"
	@echo "  check            Check proxy is reachable"
	@echo "  langfuse-status  Check if Langfuse tracing is active on running proxy"
	@echo "  langfuse-test    Send a test request through the proxy"
	@echo ""
	@echo "Langfuse (optional): export LANGFUSE_PUBLIC_KEY=pk-lf-... LANGFUSE_SECRET_KEY=sk-lf-... then make start"

install:
	uv sync

install-langfuse:
	uv sync --group langfuse

start:
	@if [ -z "$$ANTHROPIC_API_KEY" ]; then \
		echo "Error: ANTHROPIC_API_KEY is not set"; exit 1; \
	fi
	LANGFUSE_HOST=$(LANGFUSE_HOST) uv run python proxy.py

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
	@nohup uv run python proxy.py >> proxy.log 2>&1 & echo $$! > $(PID_FILE) && echo "Started PID $$(cat $(PID_FILE))"

dashboard:
	open $(URL)/dashboard

stats:
	@RESP=$$(curl -s $(URL)/stats); \
	if [ -z "$$RESP" ]; then echo "Proxy is not running at $(URL)"; exit 1; fi; \
	echo "$$RESP" | python3 -m json.tool

rtk-stats:
	rtk gain

check:
	@curl -sf $(URL)/ > /dev/null && echo "Proxy is up at $(URL)" || echo "Proxy is not running"

langfuse-status:
	@curl -sf $(URL)/ > /dev/null 2>&1 || { echo "Proxy is not running at $(URL)"; exit 0; }
	@RESP=$$(curl -s $(URL)/admin/langfuse-status); \
	if [ -z "$$RESP" ]; then echo "langfuse-status endpoint not found"; exit 1; fi; \
	echo "$$RESP" | python3 -m json.tool

langfuse-test:
	@if [ -z "$$ANTHROPIC_API_KEY" ]; then \
		echo "Error: ANTHROPIC_API_KEY is not set"; exit 1; \
	fi
	@curl -sf $(URL)/ > /dev/null 2>&1 || { echo "Error: proxy is not running. Run 'make start' first."; exit 1; }
	@echo "Sending test request through proxy..."
	@RESP=$$(curl -s -X POST $(URL)/v1/messages \
		-H "Content-Type: application/json" \
		-H "x-api-key: $$ANTHROPIC_API_KEY" \
		-H "anthropic-version: 2023-06-01" \
		-d '{"model":"claude-haiku-4-5","max_tokens":30,"messages":[{"role":"user","content":"Say hi in one word."}]}'); \
	echo "$$RESP" | python3 -m json.tool 2>/dev/null || echo "$$RESP"
	@echo ""
	@echo "Check traces: $(LANGFUSE_HOST)"
