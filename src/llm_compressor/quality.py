"""Best-effort compression-quality judging via TypeSafe's System One API.

Fire-and-forget, sampled, and fully optional: mirrors langfuse_tracer.py's shape so a
missing API key, a slow/erroring TypeSafe call, or a DB hiccup here can never add
latency to the proxy path or raise into it. See docs.typesafe.ai/api and
docs.typesafe.ai/primitives/score.md for the verified HTTP contract this implements.
"""

import asyncio
import json
import os

import httpx

TYPESAFE_API_BASE = os.environ.get("TYPESAFE_API_BASE", "https://api.typesafe.ai")
# Fresh (cache-miss) compressions: rolled once per compress_text() call.
# Raised from 0.4 -> 1.0 (judge every fresh compression) per explicit request:
# the sampled backfill left most rows "Unjudged" with no visible way to close
# the gap other than the per-row manual button, which stays scoped to
# unjudged rows only (see qualityCell() in session_detail.html).
SAMPLE_RATE = float(os.environ.get("QUALITY_SAMPLE_RATE", "1.0"))
# Cache hits: rolled per hit *only* while the key has no verdict yet (see
# compression.py's hit branch and CompressionCache.update_quality). Cache hits
# vastly outnumber misses (~97% of traffic observed in practice), so this can
# stay well below SAMPLE_RATE and still backfill the pre-existing, un-judged
# cache at a much faster rate than before without multiplying TypeSafe spend
# the way reusing SAMPLE_RATE outright would. Raised from 0.05 -> 0.25 per the
# same request as SAMPLE_RATE above.
CACHE_SAMPLE_RATE = float(os.environ.get("QUALITY_CACHE_SAMPLE_RATE", "0.25"))
THRESHOLD = 0.5  # normalized (0-1) score below this is flagged as low quality

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

    def judge(
        self,
        compression_id: int,
        original_text: str,
        compressed_text: str,
        cache_key: str | None = None,
    ) -> None:
        """Fire-and-forget sampled judging -- schedules `_judge` and discards its result."""
        if not self._client or compression_id is None:
            return
        try:
            asyncio.create_task(
                self._judge(compression_id, original_text, compressed_text, cache_key)
            )
        except Exception as exc:
            print(f"[quality] failed to schedule judgment: {exc}")

    async def judge_now(
        self,
        compression_id: int,
        original_text: str,
        compressed_text: str,
        cache_key: str | None = None,
    ) -> dict | None:
        """Awaited, on-demand judging for the per-row "Judge" button.

        Unlike `judge()` this is not sampled and its caller awaits the verdict
        directly, so the route handler can hand it straight back to the
        frontend instead of the frontend re-polling for a fire-and-forget
        result. Returns None if judging is disabled or the call/parse failed
        (persistence still happens inside `_judge` on success either way).
        """
        if not self._client or compression_id is None:
            return None
        return await self._judge(compression_id, original_text, compressed_text, cache_key)

    async def _judge(
        self,
        compression_id: int,
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
                # (q_score is None) forever, re-rolling CACHE_SAMPLE_RATE on
                # every future hit for the rest of the process's life. Patch
                # it in place so the next hit sees the verdict immediately.
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
            "sampled_pct": int(SAMPLE_RATE * 100),
            "cache_sampled_pct": int(CACHE_SAMPLE_RATE * 100),
            "api_key_set": self._api_key_set,
            "host": TYPESAFE_API_BASE,
        }


judge = QualityJudge()
