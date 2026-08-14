"""Shared design tokens and theme-toggle script, injected into every template.

Dense-analytics-console system. Pages consume roles (--series-1, --kpi-size),
never raw hex, so light/dark swap in one place.

The three categorical series colors are not free choices: they are slots 1-3 of
a palette validated with the data-viz validator under `--pairs all`, against
this file's own surfaces, in both modes. Worst all-pairs CVD deltaE is 9.2 light
/ 9.4 dark against a floor of 8; normal-vision 24.0 / 20.9 against a floor of
15. Re-run the validator before changing any --series-* value.

One caveat rides with that result: --series-3 sits at 2.82:1 on the light
surface, under the 3:1 gate. The relief rule therefore applies to every mark
drawn in it -- ship a visible direct label or a table view, never a bare swatch.
That is why the model-comparison bars carry inline values.
"""

TOKENS_CSS = """<style>
:root {
  color-scheme: dark;

  /* surfaces */
  --bg: #0c1220;
  --surface: #131b26;
  --surface-raised: #182231;
  --surface-sunken: #0e1622;
  --border: #22303f;
  --border-strong: #2e3f52;

  /* ink */
  --text: #e7edf5;
  --text-muted: #8b9bb0;
  --text-faint: #64748b;

  /* categorical series -- validated slots, fixed order, never cycled */
  --series-1: #3987e5;
  --series-2: #d95926;
  --series-3: #199e70;

  /* semantic aliases onto the series slots */
  --accent: var(--series-1);
  --accent-compressed: var(--series-1);
  --accent-saved: var(--series-3);

  /* status -- reserved, never reused as a series hue */
  --status-good: #199e70;
  --status-warn: #c98500;
  --status-critical: #e66767;
  --accent-dim: color-mix(in srgb, var(--accent) 15%, transparent);
  --good-dim: color-mix(in srgb, var(--status-good) 15%, transparent);
  --warn-dim: color-mix(in srgb, var(--status-warn) 15%, transparent);
  --critical-dim: color-mix(in srgb, var(--status-critical) 15%, transparent);

  --grid-line: color-mix(in srgb, var(--text-muted) 8%, transparent);
  --shadow-1: 0 1px 2px rgba(0, 0, 0, 0.28);

  /* type -- dense console scale */
  --font-sans: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  --font-mono: ui-monospace, "SF Mono", SFMono-Regular, Menlo, monospace;
  --fs-micro: 10px;
  --fs-sm: 11px;
  --fs-base: 13px;
  --fs-md: 15px;
  --fs-lg: 19px;
  --fs-kpi: 30px;

  /* spacing */
  --sp-1: 4px;
  --sp-2: 8px;
  --sp-3: 12px;
  --sp-4: 16px;
  --sp-5: 24px;
  --sp-6: 32px;
  --radius: 6px;
  --radius-sm: 4px;
}

/* Light values are declared twice on purpose: once under the media query for
   readers who never touched the toggle, once under [data-theme="light"] so an
   explicit choice beats the OS. Dark is the bare `:root` default above, so a
   reader on a light OS who picks dark is served by `:not([data-theme="dark"])`
   guarding the media block. */
@media (prefers-color-scheme: light) {
  :root:not([data-theme="dark"]) {
    color-scheme: light;
    --bg: #f2f4f8;
    --surface: #ffffff;
    --surface-raised: #f7f9fc;
    --surface-sunken: #eceff4;
    --border: #dde3ec;
    --border-strong: #c3ccd9;
    --text: #16202e;
    --text-muted: #5c6b80;
    --text-faint: #8593a6;
    --series-1: #2a78d6;
    --series-2: #eb6834;
    --series-3: #1baf7a;
    --status-good: #1baf7a;
    --status-warn: #b6791b;
    --status-critical: #c1453c;
    --shadow-1: 0 1px 2px rgba(16, 32, 54, 0.10);
  }
}

:root[data-theme="light"] {
  color-scheme: light;
  --bg: #f2f4f8;
  --surface: #ffffff;
  --surface-raised: #f7f9fc;
  --surface-sunken: #eceff4;
  --border: #dde3ec;
  --border-strong: #c3ccd9;
  --text: #16202e;
  --text-muted: #5c6b80;
  --text-faint: #8593a6;
  --series-1: #2a78d6;
  --series-2: #eb6834;
  --series-3: #1baf7a;
  --status-good: #1baf7a;
  --status-warn: #b6791b;
  --status-critical: #c1453c;
  --shadow-1: 0 1px 2px rgba(16, 32, 54, 0.10);
}

* { box-sizing: border-box; }

body {
  margin: 0;
  font-family: var(--font-sans);
  font-size: var(--fs-base);
  line-height: 1.45;
  background: var(--bg);
  color: var(--text);
  -webkit-font-smoothing: antialiased;
}

/* Every figure is tabular so columns of numbers align. */
.num, .mono, .tabular {
  font-family: var(--font-mono);
  font-variant-numeric: tabular-nums;
}

h1, h2, h3 { margin: 0; font-weight: 600; }

a { color: inherit; }

:focus-visible {
  outline: 2px solid var(--accent);
  outline-offset: 2px;
  border-radius: var(--radius-sm);
}

/* ---- app chrome ---- */

.nav {
  display: flex;
  align-items: center;
  gap: var(--sp-5);
  padding: 0 var(--sp-5);
  height: 46px;
  background: var(--surface);
  border-bottom: 1px solid var(--border);
  position: sticky;
  top: 0;
  z-index: 20;
}

.brand {
  display: flex;
  align-items: center;
  gap: var(--sp-2);
  font-size: var(--fs-base);
  letter-spacing: -0.01em;
  white-space: nowrap;
}
.brand b { font-weight: 650; }
.brand-mark {
  display: flex;
  align-items: flex-end;
  gap: 2px;
  height: 13px;
}
.brand-mark span {
  width: 3px;
  background: var(--accent);
  border-radius: 1px;
}
.brand-mark span:nth-child(1) { height: 13px; }
.brand-mark span:nth-child(2) { height: 9px; opacity: 0.75; }
.brand-mark span:nth-child(3) { height: 5px; opacity: 0.5; }

.nav-links { display: flex; gap: var(--sp-1); }
.nav-links a {
  padding: 5px 10px;
  border-radius: var(--radius-sm);
  color: var(--text-muted);
  text-decoration: none;
  font-size: var(--fs-base);
  font-weight: 500;
}
.nav-links a:hover { color: var(--text); background: var(--surface-raised); }
.nav-links a[aria-current="page"] { color: var(--text); background: var(--accent-dim); }

.nav-right { margin-left: auto; display: flex; align-items: center; gap: var(--sp-1); }

.icon-btn {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 28px;
  height: 28px;
  background: transparent;
  border: 1px solid transparent;
  border-radius: var(--radius-sm);
  color: var(--text-muted);
  cursor: pointer;
}
.icon-btn:hover { color: var(--text); background: var(--surface-raised); border-color: var(--border); }
.icon-btn[aria-current="page"] { color: var(--text); background: var(--accent-dim); }
.icon-btn svg { width: 15px; height: 15px; }
.icon-sun { display: none; }
:root[data-theme="light"] .icon-sun { display: block; }
:root[data-theme="light"] .icon-moon { display: none; }
@media (prefers-color-scheme: light) {
  :root:not([data-theme="dark"]) .icon-sun { display: block; }
  :root:not([data-theme="dark"]) .icon-moon { display: none; }
}

.page {
  max-width: 1360px;
  margin: 0 auto;
  padding: var(--sp-4) var(--sp-5) var(--sp-6);
}

/* breadcrumb -- the way back out of a detail view */
.crumbs {
  display: flex;
  align-items: center;
  gap: var(--sp-2);
  font-size: var(--fs-base);
  color: var(--text-muted);
  margin-bottom: var(--sp-3);
}
.crumbs a { color: var(--text-muted); text-decoration: none; }
.crumbs a:hover { color: var(--text); text-decoration: underline; }
.crumbs .sep { color: var(--text-faint); }
.crumbs .current { color: var(--text); font-weight: 550; }

.panel {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  box-shadow: var(--shadow-1);
}

.panel-head {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: var(--sp-3);
  padding: var(--sp-3) var(--sp-4);
  border-bottom: 1px solid var(--border);
}
.panel-head h2 { font-size: var(--fs-md); }

.eyebrow {
  font-size: var(--fs-sm);
  text-transform: uppercase;
  letter-spacing: 0.07em;
  color: var(--text-faint);
  font-weight: 600;
}

/* status dot -- the one liveness signal that survived; always paired with text */
.dot {
  width: 6px;
  height: 6px;
  border-radius: 50%;
  display: inline-block;
  flex: none;
  background: var(--text-faint);
}
.dot.live { background: var(--status-good); }
.dot.stale { background: var(--status-warn); }
.dot.offline { background: var(--status-critical); }

.empty {
  padding: var(--sp-5);
  text-align: center;
  color: var(--text-faint);
  font-size: var(--fs-base);
}

/* pagination -- one implementation, shared */
.pager {
  display: flex;
  align-items: center;
  gap: var(--sp-2);
  padding: var(--sp-3) var(--sp-4);
  border-top: 1px solid var(--border);
  font-size: var(--fs-base);
  color: var(--text-muted);
}
.pager button {
  background: var(--surface-raised);
  border: 1px solid var(--border);
  color: var(--text);
  border-radius: var(--radius-sm);
  padding: 3px 9px;
  cursor: pointer;
  font-size: var(--fs-base);
}
.pager button:hover:not(:disabled) { border-color: var(--border-strong); }
.pager button:disabled { opacity: 0.4; cursor: default; }
.pager .spacer { margin-left: auto; }

/* ---- layout ---- */

.row { display: flex; align-items: center; gap: var(--sp-2); }
.row.wrap { flex-wrap: wrap; }
.spread { justify-content: space-between; }
.push { margin-left: auto; }
.stack { display: flex; flex-direction: column; gap: var(--sp-3); }

/* Auto-fitting card grid. `--min` sets the narrowest a cell may get before the
   grid drops a column, so pages never hand-write breakpoints. */
.grid {
  display: grid;
  gap: var(--sp-3);
  grid-template-columns: repeat(auto-fit, minmax(var(--min, 200px), 1fr));
}
.span-2 { grid-column: span 2; }
@media (max-width: 720px) { .span-2 { grid-column: span 1; } }

/* ---- KPI tile ---- */

.kpi {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: var(--sp-3) var(--sp-4);
  box-shadow: var(--shadow-1);
  min-width: 0;
}
.kpi .label {
  font-size: var(--fs-sm);
  text-transform: uppercase;
  letter-spacing: 0.07em;
  color: var(--text-faint);
  font-weight: 600;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.kpi .value {
  font-family: var(--font-mono);
  font-variant-numeric: tabular-nums;
  font-size: var(--fs-kpi);
  line-height: 1.15;
  letter-spacing: -0.02em;
  margin-top: 2px;
}
.kpi .sub {
  font-size: var(--fs-sm);
  color: var(--text-muted);
  margin-top: 2px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.kpi.accent .value { color: var(--accent); }
.kpi.good .value { color: var(--status-good); }

/* ---- table ---- */

.table-wrap { overflow-x: auto; }

table.data {
  width: 100%;
  border-collapse: collapse;
  font-size: var(--fs-base);
}
table.data th {
  text-align: left;
  font-size: var(--fs-sm);
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: var(--text-faint);
  font-weight: 600;
  padding: var(--sp-2) var(--sp-4);
  border-bottom: 1px solid var(--border);
  white-space: nowrap;
  background: var(--surface);
  position: sticky;
  top: 0;
}
table.data td {
  padding: var(--sp-2) var(--sp-4);
  border-bottom: 1px solid var(--border);
  vertical-align: middle;
}
table.data tbody tr:last-child td { border-bottom: 0; }
table.data tbody tr:hover { background: var(--surface-raised); }
table.data td.num, table.data th.num { text-align: right; }
table.data a { text-decoration: none; font-weight: 550; }
table.data a:hover { text-decoration: underline; }
/* Whole-row link target: the cell anchor stretches, so the click area is the
   row, without nesting a block-level <a> inside every cell. */
table.data tr.linked { cursor: pointer; }

/* ---- chip / badge ---- */

.chip {
  display: inline-flex;
  align-items: center;
  gap: var(--sp-1);
  padding: 2px 7px;
  border-radius: 10px;
  border: 1px solid var(--border);
  background: var(--surface-raised);
  color: var(--text-muted);
  font-size: var(--fs-sm);
  white-space: nowrap;
}
.chip.good { border-color: transparent; background: var(--good-dim); color: var(--status-good); }
.chip.warn { border-color: transparent; background: var(--warn-dim); color: var(--status-warn); }
.chip.bad { border-color: transparent; background: var(--critical-dim); color: var(--status-critical); }
.chip.accent { border-color: transparent; background: var(--accent-dim); color: var(--accent); }

/* ---- controls ---- */

.btn {
  display: inline-flex;
  align-items: center;
  gap: var(--sp-2);
  padding: 5px 11px;
  border-radius: var(--radius-sm);
  border: 1px solid var(--border);
  background: var(--surface-raised);
  color: var(--text);
  font-family: inherit;
  font-size: var(--fs-base);
  font-weight: 500;
  cursor: pointer;
  white-space: nowrap;
}
.btn:hover:not(:disabled) { border-color: var(--border-strong); }
.btn:disabled { opacity: 0.45; cursor: default; }
.btn.primary {
  background: var(--accent);
  border-color: var(--accent);
  color: #fff;
}
.btn.primary:hover:not(:disabled) { filter: brightness(1.08); }
.btn.danger { color: var(--status-critical); }
.btn.danger:hover:not(:disabled) { border-color: var(--status-critical); }

input[type="text"], input[type="search"], input[type="number"], select, textarea {
  font-family: inherit;
  font-size: var(--fs-base);
  color: var(--text);
  background: var(--surface-sunken);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  padding: 5px 8px;
}
textarea { font-family: var(--font-mono); line-height: 1.5; resize: vertical; }
input::placeholder, textarea::placeholder { color: var(--text-faint); }

/* Segmented control -- time ranges, role filters. One row, above the charts. */
.seg {
  display: inline-flex;
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  overflow: hidden;
}
.seg button {
  border: 0;
  background: var(--surface-raised);
  color: var(--text-muted);
  font-family: inherit;
  font-size: var(--fs-sm);
  font-weight: 500;
  padding: 4px 9px;
  cursor: pointer;
  border-left: 1px solid var(--border);
}
.seg button:first-child { border-left: 0; }
.seg button:hover { color: var(--text); }
.seg button[aria-pressed="true"] { background: var(--accent-dim); color: var(--accent); }

/* Field row for settings forms. */
.field { display: flex; flex-direction: column; gap: var(--sp-1); }
.field > label { font-size: var(--fs-sm); font-weight: 600; color: var(--text-muted); }
/* Deliberately unscoped: `.field .hint` meant every hint outside a form field
   -- the model descriptions on Settings, the panel-head encoding notes --
   silently rendered as plain body text and competed with the content it was
   supposed to annotate. */
.hint { font-size: var(--fs-sm); color: var(--text-faint); }

.panel-body { padding: var(--sp-4); }
.panel-body.flush { padding: 0; }

/* ---- chart scaffolding ---- */

.legend { display: flex; flex-wrap: wrap; gap: var(--sp-3); font-size: var(--fs-sm); color: var(--text-muted); }
.legend span { display: inline-flex; align-items: center; gap: 5px; }
.legend i { width: 9px; height: 9px; border-radius: 2px; display: inline-block; }

/* Tooltip shared by every hover layer. Positioned by the page's JS. */
.tip {
  position: fixed;
  z-index: 40;
  pointer-events: none;
  display: none;
  background: var(--surface-raised);
  border: 1px solid var(--border-strong);
  border-radius: var(--radius-sm);
  box-shadow: var(--shadow-1);
  padding: var(--sp-2) var(--sp-3);
  font-size: var(--fs-sm);
  color: var(--text);
  max-width: 260px;
}
.tip .tip-title { color: var(--text-muted); margin-bottom: 3px; }
.tip .tip-row { display: flex; align-items: center; gap: var(--sp-2); }
.tip .tip-row .push { font-family: var(--font-mono); font-variant-numeric: tabular-nums; }

.sr-only {
  position: absolute;
  width: 1px; height: 1px;
  padding: 0; margin: -1px;
  overflow: hidden;
  clip: rect(0 0 0 0);
  white-space: nowrap;
  border: 0;
}

@media (max-width: 640px) {
  .nav { gap: var(--sp-3); padding: 0 var(--sp-3); }
  .nav-links a { padding: 5px 7px; }
  .page { padding: var(--sp-3) var(--sp-3) var(--sp-5); }
  :root { --fs-kpi: 24px; }
}

@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after {
    animation-duration: 0.001ms !important;
    transition-duration: 0.001ms !important;
  }
}
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
  var isLight = current === 'light'
    || (!current && window.matchMedia('(prefers-color-scheme: light)').matches);
  var next = isLight ? 'dark' : 'light';
  document.documentElement.setAttribute('data-theme', next);
  localStorage.setItem('theme', next);
  window.dispatchEvent(new CustomEvent('themechange', { detail: { theme: next } }));
};
</script>"""
