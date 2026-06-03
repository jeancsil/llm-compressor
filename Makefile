HOST  := 127.0.0.1
PORT  := 9099
URL   := http://$(HOST):$(PORT)

.PHONY: install start dashboard stats rtk-stats check help

help:
	@echo "Usage: make <target>"
	@echo ""
	@echo "  install     Install dependencies via uv"
	@echo "  start       Start the proxy (requires ANTHROPIC_API_KEY)"
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

dashboard:
	open $(URL)/dashboard

stats:
	@curl -s $(URL)/stats | python3 -m json.tool

rtk-stats:
	rtk gain

check:
	@curl -sf $(URL)/ > /dev/null && echo "Proxy is up at $(URL)" || echo "Proxy is not running"
