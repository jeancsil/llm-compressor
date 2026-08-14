import pytest


def test_render_injects_shell_tokens_and_theme_script():
    import templates

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
    import templates

    html = templates.render("sessions.html", title="Sessions", nav="sessions")

    assert '<a href="/sessions" data-nav="sessions" aria-current="page">' in html

    # The bug this whole redesign started from: every page claiming every slot,
    # or no slot at all. Exactly one nav item is current. Count inside the <nav>
    # only -- the token sheet legitimately carries `[aria-current="page"]`
    # selectors, which a whole-document count would score as extra current items.
    nav = html[html.index("<nav") : html.index("</nav>")]
    assert nav.count('aria-current="page"') == 1


def test_render_rejects_an_unknown_nav_slot():
    import templates

    with pytest.raises(ValueError, match="unknown nav slot"):
        templates.render("overview.html", title="Overview", nav="dashbaord")


def test_split_regions_separates_style_body_script():
    import templates

    style, body, script = templates._split_regions(
        "<!--#STYLE-->\n.a { color: red }\n<!--#BODY-->\n<p>hi</p>\n<!--#SCRIPT-->\nvar x = 1;\n"
    )
    assert style == ".a { color: red }"
    assert body == "<p>hi</p>"
    assert script == "var x = 1;"


def test_split_regions_tolerates_a_body_only_page():
    import templates

    style, body, script = templates._split_regions("<!--#BODY-->\n<p>hi</p>")
    assert (style, body, script) == ("", "<p>hi</p>", "")


def test_json_script_cannot_break_out_of_its_tag():
    import templates

    tag = templates.json_script("session-data", {"name": "</script><img src=x onerror=alert(1)>"})

    # The payload is inert data in an application/json block, and every `<` is
    # escaped besides -- display names are user-settable through
    # PATCH /session/{id}/name, and the old code interpolated json.dumps()
    # straight into an executable <script>.
    assert 'type="application/json"' in tag
    assert "</script><img" not in tag
    assert "\\u003c/script>" in tag or "\\u003c/script\\u003e" in tag


def test_crumbs_marks_the_last_entry_current_and_escapes_labels():
    import templates

    out = templates.crumbs(("Sessions", "/sessions"), ("<b>evil</b>", None))

    assert '<a href="/sessions">Sessions</a>' in out
    assert '<span class="current">&lt;b&gt;evil&lt;/b&gt;</span>' in out
    assert "<b>evil</b>" not in out


def test_render_places_head_extras_in_the_document_head():
    import templates

    head = templates.json_script("session-data", {"session_id": "abc"})
    html = templates.render(
        "session_detail.html", title="abc", nav="sessions", head=head, breadcrumb=""
    )

    assert head in html
    assert html.index(head) < html.index("<body>")
