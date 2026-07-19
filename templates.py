"""UI templates (HTML/CSS/JS) shipped in templates/, loaded once at import time.

Extracted from proxy.py in Task 14 (routes.py is the only current consumer:
the /dashboard, /dashboard/{session_id}, /play, and /play/list endpoints).
"""

from pathlib import Path

TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"


def _load_template(name: str) -> str:
    """Read a UI template shipped alongside the app (see templates/)."""
    return (TEMPLATES_DIR / name).read_text(encoding="utf-8")


DASHBOARD_HTML = _load_template("dashboard.html")
PLAY_HTML = _load_template("play.html")
LIST_HTML = _load_template("list.html")
