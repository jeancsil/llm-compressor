#!/usr/bin/env python3
"""Regenerate the data-driven README SVGs from the live metrics database.

The two charts in the README ("By the numbers") are rendered from real usage
logged by the proxy. They were hand-authored once and immediately went stale.
This script is the reproducible replacement:

    make assets

Reads the same database the proxy writes to (db.DB_PATH — by default
~/Library/Application Support/rtk/metrics.db on macOS), and writes:

    assets/savings-hero.svg      headline totals
    assets/savings-timeline.svg  tokens saved per day

Colors track the dashboard's dark palette so the README and the app look like
one product.
"""

import sqlite3
import sys
from pathlib import Path

from llm_compressor import db as _db

ASSETS = Path(__file__).resolve().parent.parent / "assets"

BG = "#0d1117"
CARD = "#161b22"
BORDER = "#30363d"
MUTED = "#8b949e"
TEXT = "#e6edf3"
GREEN = "#3fb950"
BLUE = "#388bfd"

SANS = "system-ui, -apple-system, sans-serif"
MONO = "'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace"

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def fetch(conn):
    """Aggregate totals plus the per-day savings series."""
    row = conn.execute(
        "SELECT SUM(original_tokens), SUM(compressed_tokens), COUNT(*) FROM compressions"
    ).fetchone()
    original, compressed, rows = row[0] or 0, row[1] or 0, row[2] or 0

    daily = conn.execute(
        "SELECT date(ts) AS d, SUM(original_tokens - compressed_tokens) AS saved "
        "FROM compressions WHERE d IS NOT NULL AND d != '' "
        "GROUP BY d ORDER BY d"
    ).fetchall()

    sessions = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]

    return {
        "original": original,
        "compressed": compressed,
        "saved": original - compressed,
        "pct": (1 - compressed / original) * 100 if original else 0.0,
        "rows": rows,
        "sessions": sessions,
        "daily": [(d, s or 0) for d, s in daily],
    }


def human(n: float) -> str:
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.1f}B"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.0f}K"
    return str(int(n))


def pretty_date(iso: str) -> str:
    _, month, day = iso.split("-")
    return f"{MONTHS[int(month) - 1]} {int(day)}"


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def text(x, y, s, size=12, fill=MUTED, family=SANS, weight=None, anchor="middle", extra=""):
    w = f' font-weight="{weight}"' if weight else ""
    return (
        f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-family="{family}" '
        f'font-size="{size}" fill="{fill}"{w}{extra}>{esc(s)}</text>'
    )


def render_hero(m) -> str:
    """Four headline stats in a single card."""
    first = pretty_date(m["daily"][0][0])
    last = pretty_date(m["daily"][-1][0])
    caption = f"LLM-Compressor · real usage data · {first} – {last} 2026"

    stats = [
        ("TOKENS SAVED", human(m["saved"]), f"of {human(m['original'])} sent", GREEN),
        ("AVG COMPRESSION", f"{m['pct']:.0f}%", "across all models", BLUE),
        ("SESSIONS", f"{m['sessions']:,}", f"{len(m['daily'])} active days", TEXT),
        ("COMPRESSIONS", human(m["rows"]), "blocks compressed", TEXT),
    ]

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="800" height="180" viewBox="0 0 800 180" role="img" aria-label="{esc(caption)}">',
        f'<rect width="800" height="180" rx="8" ry="8" fill="{CARD}" stroke="{BORDER}" stroke-width="1"/>',
        f'<line x1="0" y1="32" x2="800" y2="32" stroke="{BORDER}" stroke-width="1"/>',
        text(
            400, 21, caption, 11, MUTED, extra=' dominant-baseline="middle" letter-spacing="0.02em"'
        ),
    ]

    for i, (label, value, sub, color) in enumerate(stats):
        cx = 100 + i * 200
        parts.append(text(cx, 82, label, 10, MUTED, extra=' letter-spacing="0.08em"'))
        parts.append(text(cx, 124, value, 34, color, MONO, weight="700"))
        parts.append(text(cx, 148, sub, 10, MUTED))
        if i:
            parts.append(
                f'<line x1="{cx - 100}" y1="60" x2="{cx - 100}" y2="152" '
                f'stroke="{BORDER}" stroke-width="1"/>'
            )

    parts.append("</svg>")
    return "\n  ".join(parts) + "\n"


def render_timeline(m) -> str:
    """Bar chart of tokens saved per day."""
    daily = m["daily"]
    W, H = 800, 260
    L, R, T, B = 56, 16, 46, 46
    plot_w, plot_h = W - L - R, H - T - B

    peak = max(s for _, s in daily) or 1
    # Pick the smallest "nice" step (1/2/5 × 10^n) that keeps the axis to ~5
    # gridlines; a fixed 1M step gave 14 labels once daily peaks passed 10M.
    step = next(
        s for s in (10**e * mult for e in range(3, 12) for mult in (1, 2, 5)) if peak / s <= 5
    )
    ceiling = ((peak + step - 1) // step) * step

    bar_w = plot_w / len(daily)
    peak_day, peak_val = max(daily, key=lambda x: x[1])

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="100%" role="img" '
        f'aria-label="API-layer token savings per day">',
        f'<rect width="{W}" height="{H}" fill="{BG}"/>',
        text(W // 2, 22, "API-layer token savings per day", 13, TEXT),
    ]

    # Gridlines + y labels
    ticks = ceiling // step + 1
    for i in range(ticks):
        val = i * step
        y = T + plot_h - (val / ceiling) * plot_h
        parts.append(
            f'<line x1="{L}" y1="{y:.1f}" x2="{W - R}" y2="{y:.1f}" '
            f'stroke="{BORDER}" stroke-width="1" opacity="0.5"/>'
        )
        parts.append(text(L - 8, y + 4, human(val) if val else "0", 10, MUTED, anchor="end"))

    # Bars
    for i, (d, saved) in enumerate(daily):
        h = (saved / ceiling) * plot_h
        x = L + i * bar_w
        y = T + plot_h - h
        color = GREEN if d != peak_day else BLUE
        parts.append(
            f'<rect x="{x + bar_w * 0.15:.1f}" y="{y:.1f}" width="{bar_w * 0.7:.1f}" '
            f'height="{max(h, 0.5):.1f}" fill="{color}" rx="1"><title>{esc(d)}: '
            f"{human(saved)} tokens saved</title></rect>"
        )

    # X labels — roughly every 8th day, so they never collide.
    every = max(1, len(daily) // 8)
    for i in range(0, len(daily), every):
        x = L + i * bar_w + bar_w / 2
        parts.append(text(x, H - B + 18, pretty_date(daily[i][0]), 9, MUTED))

    # Footer summary
    parts.append(
        text(
            L, H - 10, f"peak {pretty_date(peak_day)} · {human(peak_val)}", 10, BLUE, anchor="start"
        )
    )
    parts.append(
        text(
            W - R,
            H - 10,
            f"{human(m['saved'])} tokens saved · avg {m['pct']:.1f}% · {m['sessions']:,} sessions",
            10,
            MUTED,
            anchor="end",
        )
    )

    parts.append("</svg>")
    return "\n  ".join(parts) + "\n"


def main() -> None:
    path = _db.DB_PATH
    if not Path(path).exists():
        sys.exit(f"No metrics database at {path}")

    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        m = fetch(conn)
    finally:
        conn.close()

    if not m["daily"]:
        sys.exit("No compression rows to chart yet.")

    (ASSETS / "savings-hero.svg").write_text(render_hero(m))
    (ASSETS / "savings-timeline.svg").write_text(render_timeline(m))

    print(f"Read {path}")
    print(f"  {human(m['saved'])} saved / {human(m['original'])} sent ({m['pct']:.1f}%)")
    print(
        f"  {m['sessions']:,} sessions · {len(m['daily'])} active days · {m['rows']:,} compressions"
    )
    print("Wrote assets/savings-hero.svg, assets/savings-timeline.svg")


if __name__ == "__main__":
    main()
