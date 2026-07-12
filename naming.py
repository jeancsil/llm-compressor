"""Session topic naming: deterministic extraction + heuristic/Haiku label."""

import re

_STRIP = [
    re.compile(r"<system-reminder>.*?</system-reminder>", re.S),
    re.compile(r"<local-command-[^>]*>.*?</local-command-[^>]*>", re.S),
    re.compile(r"<command-[^>]*>.*?</command-[^>]*>", re.S),
    re.compile(r"</?session>", re.S),
    re.compile(
        r"Write the title in the language the user wrote in, "
        r"regardless of the language of the examples above\.",
    ),
]


def clean(text: str) -> str:
    for rx in _STRIP:
        text = rx.sub(" ", text)
    return " ".join(text.split())


def clean_signal(
    user_turns: list[str],
    min_chars: int = 400,
    max_turns: int = 3,
    max_send: int = 3000,
) -> str:
    acc: list[str] = []
    total = 0
    for turn in user_turns[:max_turns]:
        c = clean(turn)
        if len(c) < 8 or c.startswith("/"):
            continue
        acc.append(c)
        total += len(c)
        if total >= min_chars:
            break
    return " ".join(acc)[:max_send]


_STOPWORDS = {
    "the", "a", "an", "to", "of", "in", "on", "for", "and", "or", "with",
    "please", "can", "you", "help", "me", "i", "we", "so", "that", "this",
    "my", "it", "is", "be", "add", "need", "want", "would", "like",
}


def finalize_slug(raw: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", (raw or "").lower()).strip("-")
    slug = re.sub(r"-{2,}", "-", slug)
    if not slug or slug == "general":
        return ""
    return slug


def heuristic_topic(signal: str) -> str:
    """Deterministic 2-4 word kebab topic. Empty string = no clear task (defer)."""
    first_line = (signal or "").strip().splitlines()[0] if signal.strip() else ""
    words = re.findall(r"[A-Za-z0-9]+", first_line.lower())
    keep = [w for w in words if w not in _STOPWORDS and len(w) > 1]
    if not keep:
        return ""
    return finalize_slug("-".join(keep[:4]))
