"""Shared design tokens and theme-toggle script, injected into every template.

templates.py has no include mechanism (pure string concatenation, zero build
step) -- each template embeds these via the `<!--TOKENS-->` / `<!--THEME_SCRIPT-->`
placeholders, substituted once at import time in `_load_template`.
"""

TOKENS_CSS = """<style>
:root {
  --bg: #0c1220;
  --surface: #131b26;
  --surface-raised: #182231;
  --border: #22303f;
  --text: #e7edf5;
  --text-muted: #8b9bb0;
  --accent-active: #58a6ff;
  --accent-savings: #4cbb6c;
  --accent-cache: #3fc9b0;
  --accent-warn: #e0625a;
}
@media (prefers-color-scheme: light) {
  :root {
    --bg: #eef1f6;
    --surface: #ffffff;
    --surface-raised: #f4f6fa;
    --border: #d8dfe8;
    --text: #16202e;
    --text-muted: #5c6b80;
    --accent-active: #1f6feb;
    --accent-savings: #24955a;
    --accent-cache: #128f80;
    --accent-warn: #c1453c;
  }
}
:root[data-theme="light"] {
  --bg: #eef1f6;
  --surface: #ffffff;
  --surface-raised: #f4f6fa;
  --border: #d8dfe8;
  --text: #16202e;
  --text-muted: #5c6b80;
  --accent-active: #1f6feb;
  --accent-savings: #24955a;
  --accent-cache: #128f80;
  --accent-warn: #c1453c;
}
:root[data-theme="dark"] {
  --bg: #0c1220;
  --surface: #131b26;
  --surface-raised: #182231;
  --border: #22303f;
  --text: #e7edf5;
  --text-muted: #8b9bb0;
  --accent-active: #58a6ff;
  --accent-savings: #4cbb6c;
  --accent-cache: #3fc9b0;
  --accent-warn: #e0625a;
}
body {
  font-family: -apple-system, "Segoe UI", Roboto, sans-serif;
  background: var(--bg);
  color: var(--text);
}
.mono, .tabular {
  font-family: ui-monospace, "SF Mono", Menlo, monospace;
  font-variant-numeric: tabular-nums;
}
.theme-toggle {
  background: var(--surface-raised);
  border: 1px solid var(--border);
  color: var(--text);
  border-radius: 6px;
  padding: 4px 10px;
  cursor: pointer;
  font-size: 0.85rem;
}
.theme-toggle:hover { border-color: var(--accent-active); }
</style>"""

THEME_TOGGLE_SCRIPT = """<script>
(function () {
  var saved = localStorage.getItem('theme');
  if (saved === 'light' || saved === 'dark') {
    document.documentElement.setAttribute('data-theme', saved);
  }
})();
window.toggleTheme = function () {
  var current = document.documentElement.getAttribute('data-theme');
  var isLight = current === 'light' || (!current && window.matchMedia('(prefers-color-scheme: light)').matches);
  var next = isLight ? 'dark' : 'light';
  document.documentElement.setAttribute('data-theme', next);
  localStorage.setItem('theme', next);
};
</script>"""
