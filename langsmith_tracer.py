import asyncio
import os
import uuid
from datetime import datetime, timezone


class LangSmithTracer:
    def __init__(self):
        self._client = None
        self._project = None
        self._last_run_id = None

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
            print(f"[langsmith] failed to schedule trace: {exc}")

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
                tags=tags,
                project_name=self._project,
                start_time=start,
            )
            self._client.update_run(
                run_id,
                outputs={"response": response_text},
                end_time=datetime.now(timezone.utc),
            )
            self._last_run_id = run_id
        except Exception as exc:
            print(f"[langsmith] trace failed: {exc}")

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
            print(f"[langsmith] could not create task: {exc}")

    async def _add_to_dataset(self, run_inputs: dict, run_outputs: dict, dataset_name: str) -> None:
        try:
            datasets = list(self._client.list_datasets(dataset_name=dataset_name))
            if datasets:
                dataset_id = datasets[0].id
            else:
                ds = self._client.create_dataset(
                    dataset_name,
                    description="Real proxy traffic — auto-populated",
                    data_type="kv",
                )
                dataset_id = ds.id
            self._client.create_example(
                inputs=run_inputs,
                outputs=run_outputs,
                dataset_id=dataset_id,
            )
        except Exception as exc:
            print(f"[langsmith] add_to_dataset failed: {exc}")

    async def attach_feedback(
        self,
        run_id: "uuid.UUID",
        score: float,
        comment: str = "",
    ) -> None:
        if not self._client:
            return
        try:
            asyncio.create_task(self._attach_feedback(run_id, score, comment))
        except Exception as exc:
            print(f"[langsmith] could not create task: {exc}")

    async def _attach_feedback(self, run_id: "uuid.UUID", score: float, comment: str) -> None:
        try:
            self._client.create_feedback(
                run_id=run_id,
                key="quality",
                score=max(0.0, min(1.0, score)),
                comment=comment,
            )
        except Exception as exc:
            print(f"[langsmith] attach_feedback failed: {exc}")

    def status(self) -> dict:
        return {
            "enabled": self.enabled,
            "project": self._project,
            "api_key_set": bool(os.environ.get("LANGSMITH_API_KEY", "").strip()),
        }


tracer = LangSmithTracer()
