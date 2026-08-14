"""Page rendering: a shared shell plus per-page fragments.

Pages used to be three standalone documents sharing only two `str.replace()`
calls, which is why the nav existed three times and could not express which
page you were on. Everything common -- `<head>`, tokens, nav, breadcrumb,
shared JS -- now lives in `templates/_shell.html`, and a page file supplies
only its own three regions:

    <!--#STYLE-->   page-specific CSS      (optional)
    <!--#BODY-->    markup inside <main>   (required)
    <!--#SCRIPT-->  page-specific JS       (optional)

Templates are read at import time in production and re-read per request when
LLM_COMPRESSOR_RELOAD_TEMPLATES=1, so the redesign can be iterated on without
bouncing the proxy.
"""

import html
import json
import os
from pathlib import Path

from tokens import THEME_TOGGLE_SCRIPT, TOKENS_CSS

TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"

RELOAD = os.environ.get("LLM_COMPRESSOR_RELOAD_TEMPLATES") == "1"

#: Nav slots a page may claim; `render(nav=...)` is checked against this so a
#: typo surfaces at request time instead of silently highlighting nothing.
NAV_SLOTS = frozenset({"overview", "sessions", "playground", "settings", ""})

_cache: dict[str, str] = {}


def _read(name: str) -> str:
    if RELOAD or name not in _cache:
        _cache[name] = (TEMPLATES_DIR / name).read_text(encoding="utf-8")
    return _cache[name]


def _split_regions(raw: str) -> tuple[str, str, str]:
    """Split a page file into (style, body, script) on its region markers."""
    style, script = "", ""
    body = raw
    if "<!--#STYLE-->" in body:
        _, _, rest = body.partition("<!--#STYLE-->")
        style, _, body = rest.partition("<!--#BODY-->")
    elif "<!--#BODY-->" in body:
        _, _, body = body.partition("<!--#BODY-->")
    if "<!--#SCRIPT-->" in body:
        body, _, script = body.partition("<!--#SCRIPT-->")
    return style.strip(), body.strip(), script.strip()


def json_script(element_id: str, payload: object) -> str:
    """Embed data for the page to read via `UI.bootstrap(id)`.

    A `<script type="application/json">` block, not a `<script>` assignment:
    the old code interpolated `json.dumps(session)` straight into executable
    JS, so a session named `</script><img onerror=...>` -- and display names
    are user-settable through `PATCH /session/{id}/name` -- escaped into the
    document. Here the payload is inert data, and `<` is escaped besides, so
    it cannot close the tag early.
    """
    raw = json.dumps(payload).replace("<", "\\u003c").replace("\u2028", "\\u2028")
    return (
        f'<script type="application/json" id="{html.escape(element_id, quote=True)}">{raw}</script>'
    )


def crumbs(*trail: tuple[str, str | None]) -> str:
    """Build a breadcrumb from (label, href) pairs; the last one is current."""
    parts = []
    for i, (label, href) in enumerate(trail):
        last = i == len(trail) - 1
        text = html.escape(label)
        if last or not href:
            parts.append(f'<span class="current">{text}</span>')
        else:
            parts.append(f'<a href="{html.escape(href, quote=True)}">{text}</a>')
        if not last:
            parts.append('<span class="sep" aria-hidden="true">/</span>')
    return f'<nav class="crumbs" aria-label="Breadcrumb">{"".join(parts)}</nav>'


def render(
    page: str,
    *,
    title: str,
    nav: str = "",
    breadcrumb: str = "",
    head: str = "",
) -> str:
    """Render `templates/<page>` inside the shared shell.

    `title` is the browser tab text (page name only -- the suffix is added
    here), `nav` is the slot to mark `aria-current`, `head` is extra markup
    for `<head>` such as a `json_script()` bootstrap.
    """
    if nav not in NAV_SLOTS:
        raise ValueError(f"unknown nav slot {nav!r}; expected one of {sorted(NAV_SLOTS)}")

    style, body, script = _split_regions(_read(page))
    shell = _read("_shell.html")

    page_css = f"<style>\n{style}\n</style>" if style else ""
    page_js = f"<script>\n{script}\n</script>" if script else ""
    shared_js = f"<script>\n{_read('_shared.js')}\n</script>"

    out = shell
    out = out.replace("<!--TITLE-->", html.escape(f"{title} · llm-compressor"))
    out = out.replace("<!--THEME_SCRIPT-->", THEME_TOGGLE_SCRIPT)
    out = out.replace("<!--TOKENS-->", TOKENS_CSS)
    out = out.replace("<!--HEAD-->", page_css + ("\n" + head if head else ""))
    out = out.replace("<!--CRUMBS-->", breadcrumb)
    out = out.replace("<!--BODY-->", body)
    out = out.replace("<!--SHARED_JS-->", shared_js)
    out = out.replace("<!--SCRIPT-->", page_js)

    if nav:
        out = out.replace(f'data-nav="{nav}"', f'data-nav="{nav}" aria-current="page"', 1)
    return out
