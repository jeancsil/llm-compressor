"""Tests that Makefile lifecycle targets work correctly.

These tests parse Makefile shell fragments so regressions like
"nohup VAR=val cmd" (treated as a filename by nohup) are caught before
they waste debugging time in production.
"""
import os
import re
import subprocess
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
MAKEFILE = REPO_ROOT / "Makefile"


def _makefile_text() -> str:
    return MAKEFILE.read_text()


def _restart_nohup_lines() -> list[str]:
    text = _makefile_text()
    restart_block = re.search(r"^restart:.*?(?=^\S|\Z)", text, re.MULTILINE | re.DOTALL)
    assert restart_block, "restart: target not found in Makefile"
    return [ln.strip() for ln in restart_block.group(0).splitlines() if "nohup" in ln]


# ---------------------------------------------------------------------------
# nohup env-var forwarding (the bug that killed us)
# ---------------------------------------------------------------------------

class TestMakefileRestartTarget:
    def test_nohup_does_not_have_inline_var_assignment(self):
        """nohup VAR=val cmd is invalid — nohup treats VAR=val as a filename.
        Must use: nohup env VAR=val cmd
        Root cause: 'nohup: LANGSMITH_PROJECT=llm-compressor: No such file or directory'
        """
        for line in _restart_nohup_lines():
            bad = re.search(r"nohup\s+[A-Z_]+=\S+", line)
            assert bad is None, (
                f"'nohup VAR=val' treats VAR=val as a filename — use 'nohup env VAR=val cmd'.\n"
                f"Offending line: {line}"
            )

    def test_nohup_uses_env_to_forward_vars(self):
        """nohup env ... is the correct pattern to pass env vars through nohup."""
        for line in _restart_nohup_lines():
            assert re.search(r"nohup\s+env\b", line), (
                f"nohup line doesn't use 'env' to forward variables: {line}"
            )

    def test_restart_forwards_anthropic_api_key(self):
        for line in _restart_nohup_lines():
            assert "ANTHROPIC_API_KEY" in line, (
                "ANTHROPIC_API_KEY not forwarded in restart nohup line"
            )

    def test_restart_forwards_langfuse_host(self):
        for line in _restart_nohup_lines():
            assert "LANGFUSE_HOST" in line, (
                "LANGFUSE_HOST not forwarded in restart nohup line"
            )

    def test_restart_forwards_langfuse_keys(self):
        for line in _restart_nohup_lines():
            assert "LANGFUSE_PUBLIC_KEY" in line and "LANGFUSE_SECRET_KEY" in line, (
                "LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY not forwarded in restart nohup line"
            )


# ---------------------------------------------------------------------------
# stats target must not crash when proxy is down
# ---------------------------------------------------------------------------

class TestMakefileStatsTarget:
    def test_stats_captures_before_json_parse(self):
        """Piping curl directly to json.tool crashes on empty input when proxy is down."""
        text = _makefile_text()
        stats_block = re.search(r"^stats:.*?(?=^\S|\Z)", text, re.MULTILINE | re.DOTALL)
        assert stats_block, "stats: target not found"
        recipe = stats_block.group(0)
        assert not re.search(r"curl[^;|]*\|\s*python3\s+-m\s+json\.tool", recipe), (
            "stats pipes curl directly to json.tool — crashes on empty response. "
            "Capture into a variable first."
        )

    def test_stats_exits_without_traceback(self):
        """make stats must never print a Python traceback regardless of proxy state."""
        result = subprocess.run(
            ["make", "stats"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
        )
        combined = result.stdout + result.stderr
        assert "json.decoder.JSONDecodeError" not in combined
        assert "Traceback" not in combined


# ---------------------------------------------------------------------------
# langfuse targets safe when proxy is down
# ---------------------------------------------------------------------------

class TestMakefileLangfuseTargets:
    def _target_recipe(self, name: str) -> str:
        text = _makefile_text()
        block = re.search(rf"^{name}:.*?(?=^\S|\Z)", text, re.MULTILINE | re.DOTALL)
        assert block, f"{name}: target not found in Makefile"
        return block.group(0)

    def test_langfuse_status_checks_proxy_first(self):
        recipe = self._target_recipe("langfuse-status")
        assert "curl -sf" in recipe and ("exit 0" in recipe or "exit 1" in recipe), (
            "langfuse-status doesn't guard against proxy being down"
        )

    def test_langfuse_status_exits_zero_when_proxy_down(self):
        result = subprocess.run(
            ["make", "langfuse-status"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
        )
        combined = result.stdout + result.stderr
        assert "Traceback" not in combined
        assert "json.decoder.JSONDecodeError" not in combined

    def test_langfuse_test_exits_with_message_when_proxy_down(self):
        result = subprocess.run(
            ["make", "langfuse-test"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
            env={**os.environ, "ANTHROPIC_API_KEY": "test-key"},
        )
        combined = result.stdout + result.stderr
        assert "Traceback" not in combined
        assert "json.decoder.JSONDecodeError" not in combined
        assert len(combined.strip()) > 0


# ---------------------------------------------------------------------------
# stop: no crash when PID file absent
# ---------------------------------------------------------------------------

class TestMakefileStopTarget:
    def test_stop_recipe_handles_missing_pid_file(self):
        """Static check: stop recipe must have an else branch for missing PID file.
        We don't execute make stop in tests — it kills the live proxy.
        """
        text = _makefile_text()
        stop_block = re.search(r"^stop:.*?(?=^\S|\Z)", text, re.MULTILINE | re.DOTALL)
        assert stop_block, "stop: target not found"
        recipe = stop_block.group(0)
        assert "else" in recipe, "stop target has no else branch — crashes when .proxy.pid is absent"
        assert "nothing to stop" in recipe or "not found" in recipe, (
            "stop target doesn't print a helpful message when PID file is absent"
        )


# ---------------------------------------------------------------------------
# check: always exits 0
# ---------------------------------------------------------------------------

class TestMakefileCheckTarget:
    def test_check_exits_zero_regardless_of_proxy_state(self):
        result = subprocess.run(
            ["make", "check"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0

    def test_check_prints_recognizable_status(self):
        result = subprocess.run(
            ["make", "check"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
        )
        combined = result.stdout + result.stderr
        assert any(p in combined for p in ["Proxy is up", "not running"])


# ---------------------------------------------------------------------------
# shell syntax check on the nohup env line
# ---------------------------------------------------------------------------

class TestMakefileRestartShellSyntax:
    def test_nohup_env_line_is_valid_shell(self):
        lines = _restart_nohup_lines()
        assert lines, "No nohup line in restart recipe"
        script = "\n".join([
            "#!/bin/bash",
            "ANTHROPIC_API_KEY=test",
            "ANTHROPIC_BASE_URL=http://localhost",
            "LANGFUSE_HOST=test",
            "LANGFUSE_PUBLIC_KEY=test",
            "LANGFUSE_SECRET_KEY=test",
        ] + lines)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as f:
            f.write(script)
            tmp = f.name
        try:
            result = subprocess.run(["bash", "-n", tmp], capture_output=True, text=True, timeout=5)
            assert result.returncode == 0, f"Shell syntax error:\n{result.stderr}"
        finally:
            os.unlink(tmp)
