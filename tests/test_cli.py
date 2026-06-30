import subprocess
import sys
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from cli import _proxy_path, wait_for_proxy


# --- _proxy_path ---

def test_proxy_path_points_to_existing_file():
    p = _proxy_path()
    assert p.name == "proxy.py"
    assert p.exists()


# --- wait_for_proxy ---

def test_wait_for_proxy_returns_true_when_healthy(tmp_path):
    """Spin a tiny stdlib HTTP server to simulate a healthy proxy."""
    import http.server
    import socketserver

    class _Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"status":"ok"}')

        def log_message(self, *a):
            pass  # suppress output

    with socketserver.TCPServer(("127.0.0.1", 0), _Handler) as srv:
        port = str(srv.server_address[1])
        t = threading.Thread(target=srv.handle_request)
        t.start()
        result = wait_for_proxy(port=port, timeout=5.0)
        t.join()

    assert result is True


def test_wait_for_proxy_returns_false_on_timeout():
    # Port 1 is system-reserved and will always refuse connections
    result = wait_for_proxy(port="1", timeout=0.6)
    assert result is False


# --- main() usage errors ---

def _run_cli(*args):
    return subprocess.run(
        [sys.executable, str(_proxy_path().parent / "cli.py"), *args],
        capture_output=True,
        text=True,
    )


def test_main_no_args_prints_usage():
    r = _run_cli()
    assert r.returncode == 1
    assert "Usage: llm-compressor wrap" in r.stdout


def test_main_wrong_subcommand_prints_usage():
    r = _run_cli("notvalid", "claude")
    assert r.returncode == 1
    assert "Usage: llm-compressor wrap" in r.stdout


# --- main() happy-path (fully mocked) ---

def test_main_starts_proxy_and_runs_agent():
    """Full happy-path: proxy starts, becomes healthy, agent exits 0."""
    proxy_mock = MagicMock()
    proxy_mock.poll.return_value = None  # proxy still running

    agent_result = MagicMock()
    agent_result.returncode = 0

    with (
        patch("cli.subprocess.Popen", return_value=proxy_mock) as mock_popen,
        patch("cli.wait_for_proxy", return_value=True),
        patch("cli.subprocess.run", return_value=agent_result) as mock_run,
        patch("sys.argv", ["llm-compressor", "wrap", "echo", "hello"]),
        patch("sys.exit") as mock_exit,
    ):
        import cli
        cli.main()

    mock_popen.assert_called_once()
    mock_run.assert_called_once()
    call_env = mock_run.call_args[1]["env"]
    assert call_env["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:9099"
    mock_exit.assert_called_once_with(0)


def test_main_proxy_unhealthy_exits_1():
    proxy_mock = MagicMock()
    proxy_mock.poll.return_value = None

    with (
        patch("cli.subprocess.Popen", return_value=proxy_mock),
        patch("cli.wait_for_proxy", return_value=False),
        patch("sys.argv", ["llm-compressor", "wrap", "claude"]),
        patch("sys.exit") as mock_exit,
    ):
        import cli
        try:
            cli.main()
        except SystemExit:
            pass

    mock_exit.assert_called_with(1)


def test_main_agent_not_found_exits_1():
    proxy_mock = MagicMock()
    proxy_mock.poll.return_value = None

    with (
        patch("cli.subprocess.Popen", return_value=proxy_mock),
        patch("cli.wait_for_proxy", return_value=True),
        patch("cli.subprocess.run", side_effect=FileNotFoundError),
        patch("sys.argv", ["llm-compressor", "wrap", "nonexistent-cmd"]),
        patch("sys.exit") as mock_exit,
    ):
        import cli
        try:
            cli.main()
        except SystemExit:
            pass

    mock_exit.assert_called_with(1)
