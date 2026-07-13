"""Session topic naming: deterministic extraction + heuristic/Haiku label."""

import asyncio
import re

import httpx

import sessions

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
    "the",
    "a",
    "an",
    "to",
    "of",
    "in",
    "on",
    "for",
    "and",
    "or",
    "with",
    "please",
    "can",
    "you",
    "help",
    "me",
    "i",
    "we",
    "so",
    "that",
    "this",
    "my",
    "it",
    "is",
    "be",
    "add",
    "need",
    "want",
    "would",
    "like",
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


HAIKU_MODEL = "claude-haiku-4-5-20251001"
_CLAUDE_CODE_IDENTITY = "You are Claude Code, Anthropic's official CLI for Claude."
NAMING_SYSTEM = (
    "You label developer coding sessions. Given the user's opening messages, "
    "output a 2-4 word topic in kebab-case that captures the task. Only lowercase "
    "letters, digits, and hyphens. No explanation, no quotes, no trailing period. "
    "Prefer verb-noun (fix-dashboard-css, add-batch-endpoint, debug-cache-eviction). "
    "If there is no clear task yet, output general."
)
_ANTHROPIC_MESSAGES = "https://api.anthropic.com/v1/messages"


async def haiku_topic(signal: str, auth_headers: dict) -> str:
    """One Claude-Code-shaped Haiku call on the forwarded OAuth token. Never raises."""
    try:
        headers = dict(auth_headers)
        headers["content-type"] = "application/json"
        body = {
            "model": HAIKU_MODEL,
            "max_tokens": 16,
            "system": [
                {"type": "text", "text": _CLAUDE_CODE_IDENTITY},
                {"type": "text", "text": NAMING_SYSTEM},
            ],
            "messages": [{"role": "user", "content": signal}],
        }
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(_ANTHROPIC_MESSAGES, headers=headers, json=body)
        if resp.status_code >= 400:
            print(f"[naming] haiku call {resp.status_code}")
            return ""
        text = (resp.json().get("content") or [{}])[0].get("text", "")
        return finalize_slug(text)
    except Exception as exc:  # never raise into the caller
        print(f"[naming] haiku_topic failed: {exc}")
        return ""


async def generate_topic(signal: str, auth_headers: dict, use_llm: bool) -> str:
    if use_llm:
        slug = await haiku_topic(signal, auth_headers)
        if slug:
            return slug
    return heuristic_topic(signal)


def schedule_naming(conn, session_id, signal, auth_headers, use_llm) -> None:
    """Fire-and-forget. Never blocks the request path, never raises.

    Precondition: the caller has already won `sessions.claim_for_naming`, so the
    row is in 'naming'. On success we set the auto name; on empty/failed naming we
    revert to 'provisional' so a later turn can retry (otherwise the row is stuck).
    """

    async def _run():
        try:
            slug = await generate_topic(signal, auth_headers, use_llm)
            if slug:
                sessions.apply_auto_name(conn, session_id, slug)
            else:
                sessions.revert_naming(conn, session_id)
        except Exception as exc:
            print(f"[naming] schedule_naming failed: {exc}")
            sessions.revert_naming(conn, session_id)

    try:
        asyncio.create_task(_run())
    except RuntimeError:
        # No running loop (e.g. sync test context) — run it inline, still swallowing errors.
        asyncio.run(_run())
