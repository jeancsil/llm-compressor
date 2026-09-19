import pytest


def test_render_injects_shell_tokens_and_theme_script():
    from llm_compressor import templates

    html = templates.render("overview.html", title="Overview", nav="overview")

    # Every placeholder the shell declares must be substituted, not shipped.
    for marker in (
        "<!--TITLE-->",
        "<!--TOKENS-->",
        "<!--THEME_SCRIPT-->",
        "<!--HEAD-->",
        "<!--CRUMBS-->",
        "<!--BODY-->",
        "<!--SHARED_JS-->",
        "<!--SCRIPT-->",
    ):
        assert marker not in html, f"{marker} survived rendering"

    assert "<title>Overview · llm-compressor</title>" in html
    assert "--series-1" in html  # design tokens made it in
    assert "window.toggleTheme" in html
    assert "window.UI" in html  # shared JS made it in


def test_render_marks_only_the_current_nav_item():
    from llm_compressor import templates

    html = templates.render("sessions.html", title="Sessions", nav="sessions")

    assert '<a href="/sessions" data-nav="sessions" aria-current="page">' in html

    # The bug this whole redesign started from: every page claiming every slot,
    # or no slot at all. Exactly one nav item is current. Count inside the <nav>
    # only -- the token sheet legitimately carries `[aria-current="page"]`
    # selectors, which a whole-document count would score as extra current items.
    nav = html[html.index("<nav") : html.index("</nav>")]
    assert nav.count('aria-current="page"') == 1


def test_render_rejects_an_unknown_nav_slot():
    from llm_compressor import templates

    with pytest.raises(ValueError, match="unknown nav slot"):
        templates.render("overview.html", title="Overview", nav="dashbaord")


def test_split_regions_separates_style_body_script():
    from llm_compressor import templates

    style, body, script = templates._split_regions(
        "<!--#STYLE-->\n.a { color: red }\n<!--#BODY-->\n<p>hi</p>\n<!--#SCRIPT-->\nvar x = 1;\n"
    )
    assert style == ".a { color: red }"
    assert body == "<p>hi</p>"
    assert script == "var x = 1;"


def test_split_regions_tolerates_a_body_only_page():
    from llm_compressor import templates

    style, body, script = templates._split_regions("<!--#BODY-->\n<p>hi</p>")
    assert (style, body, script) == ("", "<p>hi</p>", "")


def test_json_script_cannot_break_out_of_its_tag():
    from llm_compressor import templates

    tag = templates.json_script("session-data", {"name": "</script><img src=x onerror=alert(1)>"})

    # The payload is inert data in an application/json block, and every `<` is
    # escaped besides -- display names are user-settable through
    # PATCH /session/{id}/name, and the old code interpolated json.dumps()
    # straight into an executable <script>.
    assert 'type="application/json"' in tag
    assert "</script><img" not in tag
    assert "\\u003c/script>" in tag or "\\u003c/script\\u003e" in tag


def test_crumbs_marks_the_last_entry_current_and_escapes_labels():
    from llm_compressor import templates

    out = templates.crumbs(("Sessions", "/sessions"), ("<b>evil</b>", None))

    assert '<a href="/sessions">Sessions</a>' in out
    assert '<span class="current">&lt;b&gt;evil&lt;/b&gt;</span>' in out
    assert "<b>evil</b>" not in out


def test_session_detail_quality_chip_marks_middle_level_as_warn():
    """The middle judged level ("Drifted", rounded score 1) must render as
    `chip warn`, distinct from the top level's `chip good` -- see
    qualityChip() in session_detail.html. Only the bottom level (rounded
    score 0, low-confidence/flagged) escalates further to `chip bad`."""
    from llm_compressor import templates

    html = templates.render(
        "session_detail.html", title="x", nav="sessions", head="", breadcrumb=""
    )

    chip_fn = html[html.index("function qualityChip") : html.index("function qualityInfoButton")]
    assert "Math.round(c.score) === 1" in chip_fn
    assert "'chip warn'" in chip_fn or 'class="chip warn"' in chip_fn
    assert "Drifted" in chip_fn


def test_render_places_head_extras_in_the_document_head():
    from llm_compressor import templates

    head = templates.json_script("session-data", {"session_id": "abc"})
    html = templates.render(
        "session_detail.html", title="abc", nav="sessions", head=head, breadcrumb=""
    )

    assert head in html
    assert html.index(head) < html.index("<body>")


# ---------------------------------------------------------------------------
# Jev gate UI (Task 5)
#
# These pages are JS-rendered SPAs: the server ships a static shell plus an
# inlined <script> that fetches JSON and builds markup client-side.
# templates.render() never executes that script, so — same as the existing
# qualityChip()/qualityInfoButton() slicing tests above — these assert against
# the *source text* of the shipped page (static body copy and/or literal JS
# string constants), not against data-bound output. The route-level tests
# below (using the `client` fixture) cover the actual JSON payloads.
# ---------------------------------------------------------------------------


def test_overview_shows_gate_panel():
    from llm_compressor import templates

    html = templates.render("overview.html", title="Overview", nav="overview")

    assert "gated" in html.lower() or "served original" in html.lower()
    # The tile must be bound to the real stats field (Task 4's gated_count),
    # and the note must say headline savings already exclude gated rows.
    assert "gated_count" in html
    assert "already exclude" in html.lower()


def test_settings_shows_gate_status():
    from llm_compressor import templates

    html = templates.render("settings.html", title="Settings", nav="settings")

    assert "Lost" in html and "Drifted" in html  # block levels
    assert "gate" in html.lower()
    # States plainly that every compression is judged now (no more sampling).
    assert "every compression is judged" in html.lower()


def test_session_detail_marks_gated_rows():
    from llm_compressor import templates

    html = templates.render(
        "session_detail.html", title="x", nav="sessions", head="", breadcrumb=""
    )

    cell_fn = html[html.index("function gatedBadge") : html.index("function isExpanded")]
    assert "gated" in cell_fn.lower()
    assert "c.gated" in cell_fn
    assert "qualityCell" in html and "gatedBadge(c)" in html


def test_sessions_table_has_gated_column():
    from llm_compressor import templates

    html = templates.render("sessions.html", title="Sessions", nav="sessions")

    assert "gated" in html.lower()


def test_playground_shows_jev_verdict():
    from llm_compressor import templates

    html = templates.render("playground.html", title="Playground", nav="playground")

    assert "jev" in html.lower()
    assert "gated" in html.lower()


def test_session_compressions_endpoint_includes_gated_field(client):
    """routes.py's /session/{id}/compressions listing must carry `gated` per
    row (Task 5's routes.py change), not just the quality chip fields."""
    from llm_compressor import sessions

    sessions.record_compression(
        "s1", 100, 40, gated=1, original_text="x" * 300, compressed_text="y"
    )
    sessions.record_compression(
        "s1", 100, 40, gated=0, original_text="x" * 300, compressed_text="y"
    )

    r = client.get("/session/s1/compressions")
    assert r.status_code == 200
    items = r.json()["items"]
    assert len(items) == 2
    gated_values = {item["gated"] for item in items}
    assert gated_values == {0, 1}


def test_play_compress_includes_jev_verdict_when_enabled(client, monkeypatch):
    """playground compress response must carry the Jev verdict + gate
    decision when judging is enabled (Task 5's routes.py change)."""
    from unittest.mock import AsyncMock

    from llm_compressor import quality

    monkeypatch.setattr(quality.judge, "_client", object())  # enabled=True
    verdict = {
        "score": 1.0,
        "label": "Drifted",
        "detail": "some drift",
        "confidence": 0.7,
        "flagged": 0,
        "meta": {},
    }
    monkeypatch.setattr(quality.judge, "judge_now", AsyncMock(return_value=verdict))

    r = client.post("/play/compress", json={"text": "word " * 50, "model": ""})
    assert r.status_code == 200
    jev = r.json()["jev"]
    assert jev["label"] == "Drifted"
    assert jev["gated"] is True


def test_play_compress_omits_jev_verdict_when_disabled(client):
    """quality.judge is disabled by default in the `client` fixture (no
    TYPESAFE_API_KEY set) -- the playground response must not claim a
    verdict that was never obtained."""
    r = client.post("/play/compress", json={"text": "word " * 50, "model": ""})
    assert r.status_code == 200
    assert r.json()["jev"] is None
