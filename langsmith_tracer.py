import asyncio
import os
import uuid
from datetime import datetime, timezone


class LangSmithTracer:
    def __init__(self):
        self._client = None
        self._project = None

    def init(self, project: str = "llm-compressor") -> None:
        key = os.environ.get("LANGSMITH_API_KEY", "").strip()
        if not key:
            return
        try:
            from langsmith import Client
            self._client = Client(api_key=key)
            self._project = project
        except ImportError:
            print("[langsmith] langsmith package not installed — tracing disabled")

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
            ))
        except Exception as exc:
            print(f"[langsmith] failed to trace: {exc}")

    async def _send(
        self,
        original_messages: list,
        compressed_messages: list,
        original_system,
        compressed_system,
        response_text: str,
        metadata: dict,
    ) -> None:
        run_id = uuid.uuid4()
        start = datetime.now(timezone.utc)
        try:
            self._client.create_run(
                id=run_id,
                name="proxy-request",
                run_type="llm",
                inputs={
                    "original_messages": original_messages,
                    "compressed_messages": compressed_messages,
                    "original_system": original_system,
                    "compressed_system": compressed_system,
                },
                extra={"metadata": metadata},
                project_name=self._project,
                start_time=start,
            )
            self._client.update_run(
                run_id,
                outputs={"response": response_text},
                end_time=datetime.now(timezone.utc),
            )
        except Exception as exc:
            print(f"[langsmith] trace failed: {exc}")


tracer = LangSmithTracer()
