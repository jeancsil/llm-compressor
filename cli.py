import atexit
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

PORT = "9099"
_PROXY_URL = f"http://127.0.0.1:{PORT}"


def _proxy_path() -> Path:
    return Path(__file__).parent / "proxy.py"


def wait_for_proxy(port: str = PORT, timeout: float = 30.0) -> bool:
    url = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            urllib.request.urlopen(url, timeout=1)
            return True
        except (urllib.error.URLError, OSError):
            time.sleep(0.5)
    return False


def main() -> None:
    args = sys.argv[1:]
    if len(args) < 2 or args[0] != "wrap":
        print("Usage: llm-compressor wrap <command> [args...]")
        print("Example: llm-compressor wrap claude")
        sys.exit(1)

    agent_cmd = args[1:]

    # Forward all env vars the proxy needs; ANTHROPIC_BASE_URL will be overridden
    proxy_env = os.environ.copy()
    proxy_process = subprocess.Popen(
        [sys.executable, str(_proxy_path())],
        env=proxy_env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    def _shutdown_proxy():
        if proxy_process.poll() is None:
            proxy_process.terminate()
            try:
                proxy_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proxy_process.kill()

    atexit.register(_shutdown_proxy)

    if not wait_for_proxy(PORT):
        print("❌ Proxy failed to become healthy within 30 s.")
        sys.exit(1)
        return

    agent_env = os.environ.copy()
    agent_env["ANTHROPIC_BASE_URL"] = _PROXY_URL

    if "ANTHROPIC_API_KEY" not in agent_env:
        print("⚠️  Warning: ANTHROPIC_API_KEY is not set.")

    print(f"🔗 Injected ANTHROPIC_BASE_URL={_PROXY_URL}")
    print(f"🤖 Launching: {' '.join(agent_cmd)}\n")

    try:
        result = subprocess.run(agent_cmd, env=agent_env)
        sys.exit(result.returncode)
    except KeyboardInterrupt:
        pass
    except FileNotFoundError:
        print(f"❌ Error: Command '{agent_cmd[0]}' not found in PATH.")
        sys.exit(1)
        return


if __name__ == "__main__":
    main()
