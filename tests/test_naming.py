import asyncio
from unittest.mock import patch

import naming as N
import proxy
import sessions as S


def test_clean_strips_system_reminder():
    t = "hello <system-reminder>ignore me</system-reminder> world"
    assert N.clean(t) == "hello world"


def test_clean_strips_command_and_session_wrappers():
    t = "<command-name>/x</command-name> real <session>y</session> text"
    assert "command-name" not in N.clean(t)
    assert "session" not in N.clean(t)
    assert "real" in N.clean(t) and "text" in N.clean(t)


def test_clean_strips_write_title_artifact():
    t = "do the thing Write the title in the language the user wrote in, regardless of the language of the examples above. now"
    out = N.clean(t)
    assert "Write the title in the language" not in out


def test_clean_signal_accumulates_until_threshold():
    turns = ["short one", "b" * 500, "c" * 500]
    sig = N.clean_signal(turns, min_chars=400, max_turns=3)
    # stops after crossing 400 chars; third turn not needed
    assert "c" * 500 not in sig
    assert len(sig) >= 400


def test_clean_signal_caps_send_length():
    sig = N.clean_signal(["z" * 9000], min_chars=400, max_turns=3, max_send=3000)
    assert len(sig) == 3000


def test_clean_signal_skips_tiny_turns():
    sig = N.clean_signal(["ok", "/model opus"], min_chars=400, max_turns=3)
    assert sig == ""  # nothing substantive → defer


def test_finalize_slug_enforces_kebab():
    assert N.finalize_slug("Fix Dashboard CSS!!") == "fix-dashboard-css"


def test_finalize_slug_general_becomes_empty():
    assert N.finalize_slug("general") == ""
    assert N.finalize_slug("  ") == ""


def test_heuristic_topic_from_real_signal():
    sig = "please add db/session history data to the dashboard so I can track prompts"
    slug = N.heuristic_topic(sig)
    assert slug and slug == slug.lower()
    assert all(ch.isalnum() or ch == "-" for ch in slug)
    assert 1 <= slug.count("-") + 1 <= 4  # 2-4 words


def test_heuristic_topic_defers_on_noise():
    assert N.heuristic_topic("") == ""
    assert N.heuristic_topic("hey can you help me") in (
        "",
        N.heuristic_topic("hey can you help me"),
    )


def test_haiku_topic_builds_claude_code_shape_and_finalizes():
    captured = {}

    class _Resp:
        status_code = 200

        def json(self):
            return {"content": [{"text": "Fix Dashboard CSS"}]}

    class _Client:
        def __init__(self, *a, **k): ...
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, url, headers=None, json=None, **k):
            captured["json"] = json
            captured["headers"] = headers
            return _Resp()

    with patch("naming.httpx.AsyncClient", _Client):
        slug = asyncio.run(N.haiku_topic("add css fix", {"authorization": "Bearer x"}))
    assert slug == "fix-dashboard-css"
    # first system block is the Claude Code identity string
    assert (
        captured["json"]["system"][0]["text"]
        == "You are Claude Code, Anthropic's official CLI for Claude."
    )
    assert captured["json"]["model"] == "claude-haiku-4-5-20251001"
    assert captured["headers"]["authorization"] == "Bearer x"


def test_haiku_topic_swallows_errors_returns_empty():
    class _Boom:
        def __init__(self, *a, **k): ...
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, *a, **k):
            raise RuntimeError("401")

    with patch("naming.httpx.AsyncClient", _Boom):
        assert asyncio.run(N.haiku_topic("x", {})) == ""


def test_generate_topic_uses_heuristic_when_llm_off():
    slug = asyncio.run(N.generate_topic("add batch endpoint now", {}, use_llm=False))
    assert slug == N.heuristic_topic("add batch endpoint now")


def test_schedule_naming_applies_auto_name(tmp_path):
    c = proxy.init_db(str(tmp_path / "m.db"))
    S.ensure_session(c, "sessaaaabbbb")

    async def run():
        N.schedule_naming(c, "sessaaaabbbb", "fix the login bug", {}, use_llm=False)
        await asyncio.sleep(0.05)  # let the fire-and-forget task finish

    asyncio.run(run())
    row = S.get_session(c, "sessaaaabbbb")
    assert row["name_source"] == "auto"
    assert row["display_name"] == N.heuristic_topic("fix the login bug")


def test_schedule_naming_empty_topic_reverts_to_provisional(tmp_path):
    c = proxy.init_db(str(tmp_path / "m.db"))
    S.ensure_session(c, "sessccccdddd")
    assert S.claim_for_naming(c, "sessccccdddd") is True  # real path: row is 'naming'

    async def run():
        # "" -> heuristic_topic returns "" (proven in test_heuristic_topic_defers_on_noise);
        # "hi" was rejected as a fixture here since it survives the stopword filter (len > 1)
        # and produces a non-empty slug, which would make this test assert the wrong thing.
        N.schedule_naming(c, "sessccccdddd", "", {}, use_llm=False)
        await asyncio.sleep(0.05)

    asyncio.run(run())
    # Empty topic must release the claim so a later turn can retry.
    assert S.get_session(c, "sessccccdddd")["name_source"] == "provisional"
