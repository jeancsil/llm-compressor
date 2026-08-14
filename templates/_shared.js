/* Shared front-end core, injected into every page by templates.render().
 *
 * Everything here existed 2-3 times across dashboard.html / list.html /
 * play.html, with the copies having drifted apart -- three `fmt`s that
 * disagreed on the same input, two `buildPager`s that paginated differently.
 * One definition each, from here on.
 */
(function (window, document) {
  'use strict';

  var UI = {};

  /* ---- formatting ---------------------------------------------------- */

  /** Compact token/request counts: 1234 -> "1.2k", 1234567 -> "1.2M". */
  UI.fmt = function (n) {
    if (n === null || n === undefined || isNaN(n)) return '0';
    var v = Number(n);
    var sign = v < 0 ? '-' : '';
    v = Math.abs(v);
    if (v >= 1e9) return sign + (v / 1e9).toFixed(1).replace(/\.0$/, '') + 'B';
    if (v >= 1e6) return sign + (v / 1e6).toFixed(1).replace(/\.0$/, '') + 'M';
    if (v >= 1e4) return sign + Math.round(v / 1e3) + 'k';
    if (v >= 1e3) return sign + (v / 1e3).toFixed(1).replace(/\.0$/, '') + 'k';
    return sign + String(Math.round(v));
  };

  /** Full grouped number, for tooltips and detail rows where exactness matters. */
  UI.fmtExact = function (n) {
    if (n === null || n === undefined || isNaN(n)) return '0';
    return Number(n).toLocaleString('en-US');
  };

  UI.fmtPct = function (n, digits) {
    if (n === null || n === undefined || isNaN(n)) return '—';
    return Number(n).toFixed(digits === undefined ? 1 : digits) + '%';
  };

  /**
   * A duration in ms, in the largest unit that keeps it readable.
   *
   * The ladder runs past seconds because this formats cumulative totals as well
   * as per-request latency: the cache's "time saved" is the sum of every miss
   * avoided since deploy, and rendering that as "88659.68s" hands the reader an
   * arithmetic problem instead of an answer ("24.6h").
   */
  /**
   * A 0..1 fraction as a percentage.
   *
   * Distinct from fmtPct, which takes an already-scaled number. The API mixes
   * both conventions -- `avg_savings_pct` is 37.6, `hit_ratio` is 0.9604 -- and
   * feeding a ratio to fmtPct silently renders a 96% cache hit rate as "1.0%".
   * That is exactly what Settings did while Overview, doing its own `* 100`,
   * showed the truth; the same number disagreed with itself across two pages.
   */
  UI.fmtRatioPct = function (n, digits) {
    if (n === null || n === undefined || isNaN(n)) return '—';
    return UI.fmtPct(Number(n) * 100, digits);
  };

  UI.fmtMs = function (n) {
    if (n === null || n === undefined || isNaN(n)) return '—';
    var v = Number(n);
    if (v >= 86400000) return (v / 86400000).toFixed(1) + 'd';
    if (v >= 3600000) return (v / 3600000).toFixed(1) + 'h';
    if (v >= 60000) return (v / 60000).toFixed(1) + 'm';
    if (v >= 1000) return (v / 1000).toFixed(2) + 's';
    return Math.round(v) + 'ms';
  };

  UI.fmtCost = function (usd) {
    if (usd === null || usd === undefined || isNaN(usd)) return '$0.00';
    var v = Number(usd);
    if (v >= 1000) return '$' + UI.fmt(v);
    if (v < 0.01 && v > 0) return '<$0.01';
    return '$' + v.toFixed(2);
  };

  /** "3m ago" / "2h ago" / "5d ago". Returns "—" for anything unparseable. */
  UI.ago = function (ts) {
    var d = UI.parseTs(ts);
    if (!d) return '—';
    var secs = (Date.now() - d.getTime()) / 1000;
    if (secs < 0) secs = 0;
    if (secs < 60) return Math.floor(secs) + 's ago';
    if (secs < 3600) return Math.floor(secs / 60) + 'm ago';
    if (secs < 86400) return Math.floor(secs / 3600) + 'h ago';
    return Math.floor(secs / 86400) + 'd ago';
  };

  /**
   * Parse a stored timestamp. The `compressions.ts` column has historically
   * held three formats; rows written before the normalization migration may
   * still be naive, so treat a bare ISO string as UTC rather than local.
   */
  UI.parseTs = function (ts) {
    if (!ts) return null;
    var s = String(ts).trim();
    if (!s) return null;
    if (!/[Zz]|[+\-]\d{2}:?\d{2}$/.test(s)) s = s.replace(' ', 'T') + 'Z';
    var d = new Date(s);
    return isNaN(d.getTime()) ? null : d;
  };

  UI.fmtTime = function (ts) {
    var d = UI.parseTs(ts);
    if (!d) return '—';
    return d.toLocaleString(undefined, {
      month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit',
    });
  };

  /* ---- escaping ------------------------------------------------------ */

  /** Escape for text content and quoted attribute values alike. */
  UI.escapeHtml = function (s) {
    if (s === null || s === undefined) return '';
    return String(s)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');
  };

  /* ---- fetch --------------------------------------------------------- */

  UI.getJSON = function (url) {
    return fetch(url, { headers: { Accept: 'application/json' } }).then(function (r) {
      if (!r.ok) throw new Error(url + ' -> HTTP ' + r.status);
      return r.json();
    });
  };

  UI.postJSON = function (url, body, method) {
    return fetch(url, {
      method: method || 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body || {}),
    }).then(function (r) {
      return r.json().catch(function () { return {}; }).then(function (data) {
        if (!r.ok) throw new Error(data.error || data.detail || ('HTTP ' + r.status));
        return data;
      });
    });
  };

  /* ---- pagination ---------------------------------------------------- */

  /**
   * Render a pager into `el`. `state` is {page, perPage, total}; `onPage` gets
   * the new zero-based page. Renders nothing when everything fits on one page.
   */
  UI.buildPager = function (el, state, onPage) {
    if (!el) return;
    var pages = Math.max(1, Math.ceil(state.total / state.perPage));
    if (state.total === 0 || pages === 1) { el.innerHTML = ''; return; }
    var first = state.page * state.perPage + 1;
    var last = Math.min(state.total, (state.page + 1) * state.perPage);

    el.innerHTML =
      '<button type="button" data-pg="prev"' + (state.page === 0 ? ' disabled' : '') + '>Prev</button>' +
      '<button type="button" data-pg="next"' + (state.page >= pages - 1 ? ' disabled' : '') + '>Next</button>' +
      '<span class="spacer num">' + first + '–' + last + ' of ' + UI.fmtExact(state.total) + '</span>';

    el.querySelectorAll('[data-pg]').forEach(function (btn) {
      btn.addEventListener('click', function () {
        var next = btn.getAttribute('data-pg') === 'prev' ? state.page - 1 : state.page + 1;
        if (next < 0 || next >= pages) return;
        onPage(next);
      });
    });
  };

  /* ---- polling ------------------------------------------------------- */

  /**
   * One poller for the whole page.
   *
   * Session detail used to arm five uncoordinated setIntervals against four
   * endpoints, none of which backed off on failure or stopped for a hidden
   * tab -- a backgrounded dashboard kept hammering the proxy all day. This
   * runs registered tasks on one timer, pauses on `visibilitychange`, fires
   * immediately on return, and backs a failing task off exponentially to 60s.
   */
  UI.poller = (function () {
    var tasks = [];
    var timer = null;
    var BASE = 5000;
    var MAX = 60000;

    function runTask(t, now) {
      if (now < t.nextAt) return;
      t.nextAt = now + t.interval;
      Promise.resolve()
        .then(t.fn)
        .then(function () {
          t.interval = t.baseInterval;
          if (t.failing) {
            t.failing = false;
            document.dispatchEvent(new CustomEvent('poll:ok', { detail: { name: t.name } }));
          }
        })
        .catch(function (err) {
          t.failing = true;
          t.interval = Math.min(MAX, t.interval * 2);
          t.nextAt = Date.now() + t.interval;
          document.dispatchEvent(new CustomEvent('poll:fail', {
            detail: { name: t.name, error: String(err && err.message || err) },
          }));
        });
    }

    function tick() {
      if (document.hidden) return;
      var now = Date.now();
      tasks.forEach(function (t) { runTask(t, now); });
    }

    document.addEventListener('visibilitychange', function () {
      if (document.hidden) return;
      // Catch up immediately rather than waiting out the tick we skipped.
      tasks.forEach(function (t) { t.nextAt = 0; });
      tick();
    });

    return {
      /**
       * Register `fn` to run every `ms` (default 5s) and once right away.
       * Registering a name twice replaces the first task rather than stacking
       * a second one: transient tasks (a model load finishing, say) get armed
       * from an event handler that can fire repeatedly.
       */
      add: function (name, fn, ms) {
        var interval = ms || BASE;
        tasks = tasks.filter(function (t) { return t.name !== name; });
        tasks.push({
          name: name, fn: fn, baseInterval: interval, interval: interval,
          nextAt: 0, failing: false,
        });
        if (!timer) timer = setInterval(tick, 1000);
        tick();
      },
      /** Retire a finished transient task. Unknown names are a no-op. */
      remove: function (name) {
        tasks = tasks.filter(function (t) { return t.name !== name; });
      },
      /** Force every task to run now (after a mutation, say). */
      refresh: function () { tasks.forEach(function (t) { t.nextAt = 0; }); tick(); },
      stop: function () { clearInterval(timer); timer = null; tasks = []; },
    };
  }());

  /* ---- misc ---------------------------------------------------------- */

  /** Read a `<script type="application/json">` payload by element id. */
  UI.bootstrap = function (id) {
    var el = document.getElementById(id);
    if (!el) return null;
    try { return JSON.parse(el.textContent); } catch (e) { return null; }
  };

  /** Current resolved theme, for canvas/SVG code that must pick real colors. */
  UI.theme = function () {
    var set = document.documentElement.getAttribute('data-theme');
    if (set) return set;
    return window.matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark';
  };

  /** Resolve a CSS custom property to its computed value. */
  UI.cssVar = function (name) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  };

  window.UI = UI;
}(window, document));
