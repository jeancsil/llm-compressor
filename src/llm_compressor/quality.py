"""Compression-quality judging via TypeSafe's System One API, and the gate it drives.

Every unique compression is judged once, inline, by `compress_text()` in
compression.py: `judge_now()` is awaited with a bounded timeout (`GATE_TIMEOUT`)
and its verdict decides whether the compressed or the original text is
forwarded (`gate_decision()`). A missing API key, a slow/erroring TypeSafe
call, or a DB hiccup here can never add unbounded latency to the proxy path or
raise into it -- the caller always falls back to forwarding the compressed
text. See docs.typesafe.ai/api and docs.typesafe.ai/primitives/score.md for
the verified HTTP contract this implements.
"""

import json
import os

import httpx

TYPESAFE_API_BASE = os.environ.get("TYPESAFE_API_BASE", "https://api.typesafe.ai")
THRESHOLD = 0.5  # normalized (0-1) score below this is flagged as low quality
GATE_TIMEOUT = float(os.environ.get("QUALITY_GATE_TIMEOUT", "5.0"))

_CRITERIA = [
    "Meaning is lost or significantly changed — key facts, instructions, or intent "
    "from the original are missing or altered in the compressed text.",
    "Meaning is mostly preserved but with noticeable drift — some nuance, detail, or "
    "phrasing changed enough that it could affect interpretation.",
    "Meaning is fully preserved — the compressed text conveys the same facts, "
    "instructions, and intent as the original.",
]

# Short chip text for each _CRITERIA index (by rounded raw score). TypeSafe's
# `legend` gives back the full criteria sentence for the rounded level -- that
# stays available as `detail` (a tooltip) so the chip itself can show a label
# short enough to sit inline in a table cell.
SHORT_LABELS = {0: "Lost", 1: "Drifted", 2: "Preserved"}

# Verdict raw score rounds to a _CRITERIA index: 0=Lost, 1=Drifted, 2=Preserved.
# The gate blocks (forwards the original) on Lost/Drifted; only Preserved keeps
# the compressed text. Absent/malformed verdict -> allow (savings-first).
_BLOCK_LEVELS = {0, 1}


def gate_decision(raw_score: "float | None") -> str:
    if raw_score is None:
        return "allow"
    try:
        return "block" if round(float(raw_score)) in _BLOCK_LEVELS else "allow"
    except (TypeError, ValueError):
        return "allow"


class QualityJudge:
    def __init__(self):
        self._client = None
        self._api_key_set = False

    def init(self) -> None:
        key = os.environ.get("TYPESAFE_API_KEY", "").strip()
        if not key:
            return
        self._api_key_set = True
        self._client = httpx.AsyncClient(
            base_url=TYPESAFE_API_BASE,
            headers={"Authorization": f"Bearer {key}"},
            timeout=10.0,
        )
        print(f"[quality] TypeSafe judging enabled → {TYPESAFE_API_BASE}")

    @property
    def enabled(self) -> bool:
        return self._client is not None

    async def judge_now(
        self,
        compression_id: int | None,
        original_text: str,
        compressed_text: str,
        cache_key: str | None = None,
    ) -> dict | None:
        """Awaited, on-demand judging.

        Two callers:
        - The per-row "Judge" button (routes.py), with a real `compression_id`:
          the verdict is persisted to `compression_quality` as before.
        - The gate in `compress_text()` (compression.py), with
          `compression_id=None`: called BEFORE the compression row exists, so
          only the Jev call + `compression_cache` verdict columns (when
          `cache_key` is set) are wanted. `_judge` skips the
          `compression_quality` INSERT in that case -- see its body.

        Returns None if judging is disabled or the call/parse/persist failed.
        """
        if not self._client:
            return None
        return await self._judge(compression_id, original_text, compressed_text, cache_key)

    async def _judge(
        self,
        compression_id: int | None,
        original_text: str,
        compressed_text: str,
        cache_key: str | None,
    ) -> dict | None:
        try:
            resp = await self._client.post(
                "/v1/systemone",
                json={
                    "model": "jev-latest",
                    "state": {
                        "original_text": original_text,
                        "compressed_text": compressed_text,
                    },
                    "questions": {
                        "meaning_preserved": {
                            "type": "score",
                            "instructions": (
                                "How well does the compressed_text preserve the meaning "
                                "of the original_text?"
                            ),
                            "criteria": _CRITERIA,
                        }
                    },
                },
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            print(f"[quality] judgment failed: {exc}")
            return None

        try:
            answer = data["answers"]["meaning_preserved"]
            raw_score = float(answer["score"])
            confidence = float(answer.get("confidence", 0.0))
            legend = answer.get("legend") or {}
            num_levels = len(_CRITERIA)
            normalized = raw_score / (num_levels - 1) if num_levels > 1 else raw_score
            # `legend` keys may come back as strings ("0") or ints depending on the
            # real response's JSON decoding -- try both so a bare int rounding of
            # raw_score resolves to a label either way.
            rounded = round(raw_score)
            # `detail` is TypeSafe's full criteria sentence for the rounded level
            # (what used to be shown, verbatim, as the chip's own label -- too
            # long to sit inline in a table cell). `label` is the short chip text;
            # `detail` becomes its tooltip.
            detail = legend.get(str(rounded), legend.get(rounded, ""))
            label = SHORT_LABELS.get(rounded, "")
            flagged = 1 if normalized < THRESHOLD else 0
            # Everything else jev hands back that isn't already captured above
            # (per-level probabilities, token usage, the responding model
            # name) -- persisted as one JSON blob rather than a column each,
            # since it's display-only (the frontend's info-button panel) and
            # never queried on. `.get(..., {})` throughout: older/lighter test
            # fixtures and real responses alike may omit any of these.
            meta = {
                "probabilities": answer.get("probabilities") or {},
                "usage": data.get("usage") or {},
                "model": data.get("model", ""),
            }
        except Exception as exc:
            print(f"[quality] malformed response, skipping: {exc}")
            return None

        try:
            from llm_compressor import db

            if db._db_conn is None:
                return None
            # `compression_id is None` means this call came from the gate in
            # compress_text() -- BEFORE the compression row exists, so there is
            # no valid foreign key for compression_quality yet. Skip that
            # INSERT in that case; the per-row "Judge" button path (a real
            # compression_id) is unchanged and still persists it.
            if compression_id is not None:
                db._db_conn.execute(
                    "INSERT OR REPLACE INTO compression_quality "
                    "(compression_id, score, label, detail, confidence, flagged, judged_at, meta) "
                    "VALUES (?,?,?,?,?,?,?,?)",
                    (
                        compression_id,
                        raw_score,
                        label,
                        detail,
                        confidence,
                        flagged,
                        db.utc_now(),
                        json.dumps(meta),
                    ),
                )
            if cache_key is not None:
                db._db_conn.execute(
                    "UPDATE compression_cache SET quality_score=?, quality_label=?, "
                    "quality_detail=?, quality_flagged=? WHERE key=?",
                    (raw_score, label, detail, flagged, cache_key),
                )
            db._db_conn.commit()
        except Exception as exc:
            print(f"[quality] failed to persist judgment: {exc}")
            return None

        if cache_key is not None:
            try:
                # The disk row is now judged, but CompressionCache.get() only
                # reads disk on a cold miss -- an entry already resident in
                # this process's `_mem` LRU would otherwise look unjudged
                # (q_score is None) forever, re-calling Jev on every future hit
                # for the rest of the process's life. Patch it in place so the
                # next hit sees the verdict immediately and takes the
                # instant-gate branch in compress_text().
                from llm_compressor import compression

                if compression._cache is not None:
                    compression._cache.update_quality(cache_key, raw_score, label, detail, flagged)
            except Exception as exc:
                print(f"[quality] failed to refresh in-memory cache entry: {exc}")

        return {
            "score": raw_score,
            "label": label,
            "detail": detail,
            "confidence": confidence,
            "flagged": flagged,
            "meta": meta,
        }

    def status(self) -> dict:
        return {
            "enabled": self.enabled,
            "gate_enabled": self.enabled,
            "block_levels": [SHORT_LABELS[0], SHORT_LABELS[1]],  # ["Lost", "Drifted"]
            "timeout_s": GATE_TIMEOUT,
            "fallback": "compressed",
            "api_key_set": self._api_key_set,
            "host": TYPESAFE_API_BASE,
        }


judge = QualityJudge()
