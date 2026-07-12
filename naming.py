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
