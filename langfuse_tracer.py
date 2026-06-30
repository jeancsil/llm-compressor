import asyncio
import os
from datetime import datetime, timezone


class LangfuseTracer:
    def __init__(self):
        self._client = None
        self._host = os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com")
        self._last_trace_id = None

    def init(self) -> None:
        pub = os.environ.get("LANGFUSE_PUBLIC_KEY", "").strip()
        sec = os.environ.get("LANGFUSE_SECRET_KEY", "").strip()
        if not pub or not sec:
            return
        self._host = os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com")
        try:
            from langfuse import Langfuse
            self._client = Langfuse(
                public_key=pub,
                secret_key=sec,
                host=self._host,
            )
            print(f"[langfuse] tracing enabled → {self._host}")
        except ImportError:
            print("[langfuse] langfuse package not installed — tracing disabled")

    @property
    def enabled(self) -> bool:
        return self._client is not None

    async def log_request(
        self,
        original_messages: list,
        compressed_messages: list,
        original_system,
        compressed_system,
        response_text: str,
        metadata: dict,
        tags: list | None = None,
    ) -> None:
        if not self._client:
            return
        try:
            asyncio.create_task(self._send(
                original_messages,
                compressed_messages,
                original_system,
                compressed_system,
                response_text,
                metadata,
                tags or [],
            ))
        except Exception as exc:
            print(f"[langfuse] failed to schedule trace: {exc}")

    async def _send(
        self,
        original_messages: list,
        compressed_messages: list,
        original_system,
        compressed_system,
        response_text: str,
        metadata: dict,
        tags: list,
    ) -> None:
        start = datetime.now(timezone.utc)
        try:
            trace = self._client.trace(
                name="proxy-request",
                input={
                    "original_messages": original_messages,
                    "original_system": original_system,
                },
                tags=tags,
                metadata=metadata,
            )
            trace.generation(
                name="anthropic-call",
                model=metadata.get("anthropic_model", "unknown"),
                input={
                    "messages": compressed_messages,
                    "system": compressed_system,
                },
                output={"text": response_text},
                start_time=start,
                end_time=datetime.now(timezone.utc),
                metadata=metadata,
            )
            self._last_trace_id = trace.id
        except Exception as exc:
            print(f"[langfuse] trace failed: {exc}")

    async def add_to_dataset(
        self,
        run_inputs: dict,
        run_outputs: dict,
        dataset_name: str = "proxy-production-traffic",
    ) -> None:
        if not self._client:
            return
        try:
            asyncio.create_task(self._add_to_dataset(run_inputs, run_outputs, dataset_name))
        except Exception as exc:
            print(f"[langfuse] could not create task: {exc}")

    async def _add_to_dataset(self, run_inputs: dict, run_outputs: dict, dataset_name: str) -> None:
        try:
            self._client.create_dataset_item(
                dataset_name=dataset_name,
                input=run_inputs,
                expected_output=run_outputs,
            )
        except Exception as exc:
            print(f"[langfuse] add_to_dataset failed: {exc}")

    async def attach_feedback(
        self,
        trace_id: str,
        score: float,
        comment: str = "",
    ) -> None:
        if not self._client:
            return
        try:
            asyncio.create_task(self._attach_feedback(trace_id, score, comment))
        except Exception as exc:
            print(f"[langfuse] could not create task: {exc}")

    async def _attach_feedback(self, trace_id: str, score: float, comment: str) -> None:
        try:
            self._client.score(
                trace_id=trace_id,
                name="quality",
                value=max(0.0, min(1.0, score)),
                comment=comment,
            )
        except Exception as exc:
            print(f"[langfuse] attach_feedback failed: {exc}")

    def status(self) -> dict:
        return {
            "enabled": self.enabled,
            "host": self._host,
            "public_key_set": bool(os.environ.get("LANGFUSE_PUBLIC_KEY", "").strip()),
            "secret_key_set": bool(os.environ.get("LANGFUSE_SECRET_KEY", "").strip()),
        }


tracer = LangfuseTracer()
