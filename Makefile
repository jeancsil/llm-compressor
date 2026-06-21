HOST  := 127.0.0.1
PORT  := 9099
URL   := http://$(HOST):$(PORT)
PID_FILE := .proxy.pid

.PHONY: install start restart stop dashboard stats rtk-stats check help

help:
	@echo "Usage: make <target>"
	@echo ""
	@echo "  install     Install dependencies via uv"
	@echo "  start       Start the proxy (requires ANTHROPIC_API_KEY)"
	@echo "  stop        Stop the running proxy"
	@echo "  restart     Stop and start fresh (requires ANTHROPIC_API_KEY)"
	@echo "  dashboard   Open the dashboard in the browser"
	@echo "  stats       Print proxy compression stats (JSON)"
	@echo "  rtk-stats   Print rtk shell-layer savings"
	@echo "  check       Check proxy is reachable"

install:
	uv sync

start:
	@if [ -z "$$ANTHROPIC_API_KEY" ]; then \
		echo "Error: ANTHROPIC_API_KEY is not set"; exit 1; \
	fi
	uv run python llmlingua_proxy.py

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
	@nohup uv run python llmlingua_proxy.py >> proxy.log 2>&1 & echo $$! > $(PID_FILE) && echo "Started PID $$(cat $(PID_FILE))"

dashboard:
	open $(URL)/dashboard

stats:
	@curl -s $(URL)/stats | python3 -m json.tool

rtk-stats:
	rtk gain

check:
	@curl -sf $(URL)/ > /dev/null && echo "Proxy is up at $(URL)" || echo "Proxy is not running"
