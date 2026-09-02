from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass
from uuid import uuid4

import grpc
import httpx

from rag_orchestrator import RagResult, Settings
from rag_transport.proto import ml_pb2, ml_pb2_grpc

from .composition import ApplicationContainer, build_container

logger = logging.getLogger(__name__)

_MAX_GRPC_MESSAGE_BYTES = 4 << 20
_MAX_QUERY_CHARS = 16_000
_MAX_TRACE_ID_BYTES = 128
_TOKEN_SPLIT_RE = re.compile(r"\S+\s*", re.UNICODE)


@dataclass(slots=True)
class SyncJob:
    job_id: str
    status: str
    processed_docs: int = 0
    total_docs: int = 0
    error: str = ""


class RagRuntime:
    """Long-lived RAG runtime shared by all gRPC calls.

    Queries use a snapshot of the active container.  Knowledge-base sync builds a
    fresh container in the background and swaps it in atomically after success,
    so in-flight queries continue using the previous healthy index.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.container: ApplicationContainer | None = None
        self._query_slots = asyncio.Semaphore(settings.grpc_max_concurrent_queries)
        self._jobs: dict[str, SyncJob] = {}
        self._jobs_lock = asyncio.Lock()
        self._active_sync_job_id: str | None = None

    async def initialize(self) -> None:
        self.container = await asyncio.to_thread(build_container, self.settings)

    def current_container(self) -> ApplicationContainer:
        if self.container is None:
            raise RuntimeError("RAG runtime is not initialized")
        return self.container

    async def run_query(self, query: str, top_k: int, trace_id: str, section_id: int = 0) -> RagResult:
        async with self._query_slots:
            container = self.current_container()
            return await asyncio.to_thread(
                container.pipeline.run,
                query,
                top_k,
                trace_id,
                section_id,
            )

    async def start_sync(self) -> SyncJob:
        async with self._jobs_lock:
            if self._active_sync_job_id:
                active = self._jobs.get(self._active_sync_job_id)
                if active and active.status in {"pending", "running"}:
                    return active

            current_total = 0
            if self.container is not None:
                current_total = self.container.index.manifest.document_count
            job = SyncJob(
                job_id=f"kb-{uuid4().hex}",
                status="pending",
                processed_docs=0,
                total_docs=current_total,
            )
            self._jobs[job.job_id] = job
            self._active_sync_job_id = job.job_id
            self._prune_jobs_locked()
            asyncio.create_task(self._run_sync_job(job.job_id), name=f"rag-sync-{job.job_id}")
            return job

    async def _run_sync_job(self, job_id: str) -> None:
        async with self._jobs_lock:
            job = self._jobs[job_id]
            job.status = "running"

        try:
            fresh = await asyncio.to_thread(
                build_container,
                self.settings,
                force_index=True,
            )
        except Exception as exc:  # noqa: BLE001 - surfaced as job status, logged in full.
            logger.exception("Knowledge-base sync failed. job_id=%s", job_id)
            async with self._jobs_lock:
                job = self._jobs[job_id]
                job.status = "failed"
                job.error = f"{type(exc).__name__}: {exc}"[:1000]
                self._active_sync_job_id = None
            return

        # Swap only after the new index + pipeline are completely usable.
        self.container = fresh
        total = fresh.index.manifest.document_count
        async with self._jobs_lock:
            job = self._jobs[job_id]
            job.status = "completed"
            job.processed_docs = total
            job.total_docs = total
            self._active_sync_job_id = None

    async def get_sync_job(self, job_id: str) -> SyncJob | None:
        async with self._jobs_lock:
            return self._jobs.get(job_id)

    def _prune_jobs_locked(self) -> None:
        overflow = len(self._jobs) - self.settings.grpc_max_sync_jobs
        if overflow <= 0:
            return
        for key in list(self._jobs):
            if overflow <= 0:
                break
            if key == self._active_sync_job_id:
                continue
            self._jobs.pop(key, None)
            overflow -= 1


class MLGatewayService:
    def __init__(self, runtime: RagRuntime) -> None:
        self.runtime = runtime

    async def Query(self, request, context):  # noqa: N802 - protobuf RPC name
        query = str(request.query or "").strip()
        if not query:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "query must not be empty")
        if len(query) > _MAX_QUERY_CHARS:
            await context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                f"query must not exceed {_MAX_QUERY_CHARS} characters",
            )

        trace_id = _resolve_trace_id(request.trace_id, context.invocation_metadata())
        if len(trace_id.encode("utf-8")) > _MAX_TRACE_ID_BYTES:
            await context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                f"trace_id must not exceed {_MAX_TRACE_ID_BYTES} bytes",
            )

        # The RAG result is capped at five source chunks; higher values are clamped.
        top_k = max(1, min(int(request.top_k or 5), 5))
        section_id = max(0, int(getattr(request, "section_id", 0) or 0))

        try:
            result = await self.runtime.run_query(query, top_k, trace_id, section_id)
        except ValueError as exc:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
        except Exception as exc:  # noqa: BLE001
            logger.exception("RAG query failed. trace_id=%s", trace_id)
            await context.abort(grpc.StatusCode.INTERNAL, f"RAG query failed: {type(exc).__name__}")

        route_type = result.route_type.value
        yield ml_pb2.QueryChunk(
            route_info=ml_pb2.RouteInfo(route_type=route_type)
        )

        if self.runtime.settings.grpc_stream_tokens:
            for token_chunk in _chunk_answer(
                result.answer,
                self.runtime.settings.grpc_token_chunk_chars,
            ):
                if context.cancelled():
                    return
                yield ml_pb2.QueryChunk(token=token_chunk)

        pb_sources = [
            ml_pb2.SourceItem(
                doc_id=source.doc_id,
                title=source.title,
                url=source.url,
                score=float(source.score),
            )
            for source in result.sources
        ]
        yield ml_pb2.QueryChunk(
            final=ml_pb2.FinalResult(
                answer=result.answer,
                sources=pb_sources,
                route_type=route_type,
                latency_ms=max(0, int(result.latency_ms)),
            )
        )

    async def SyncKnowledgeBase(self, request, context):  # noqa: N802
        try:
            job = await self.runtime.start_sync()
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to start knowledge-base sync")
            await context.abort(grpc.StatusCode.INTERNAL, f"sync start failed: {type(exc).__name__}")
        return ml_pb2.SyncResponse(job_id=job.job_id, status=job.status)

    async def GetSyncStatus(self, request, context):  # noqa: N802
        job_id = str(request.job_id or "").strip()
        if not job_id:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "job_id is required")
        job = await self.runtime.get_sync_job(job_id)
        if job is None:
            await context.abort(grpc.StatusCode.NOT_FOUND, "sync job not found")
        return ml_pb2.SyncStatus(
            job_id=job.job_id,
            status=job.status,
            processed_docs=max(0, int(job.processed_docs)),
            total_docs=max(0, int(job.total_docs)),
        )

    async def HealthCheck(self, request, context):  # noqa: N802
        try:
            container = self.runtime.current_container()
        except RuntimeError:
            return ml_pb2.HealthCheckResponse(
                healthy=False,
                llm_status="not_initialized",
                vector_store_status="not_initialized",
            )

        manifest = container.index.manifest
        vector_status = (
            f"ready:documents={manifest.document_count},chunks={manifest.chunk_count}"
        )
        llm_ok, llm_status = await asyncio.to_thread(
            _check_llm,
            container.settings,
        )
        return ml_pb2.HealthCheckResponse(
            healthy=bool(llm_ok and manifest.chunk_count > 0),
            llm_status=llm_status,
            vector_store_status=vector_status,
        )


def _resolve_trace_id(request_trace_id: str, metadata) -> str:
    trace_id = str(request_trace_id or "").strip()
    if trace_id:
        return trace_id
    for item in metadata or ():
        key = getattr(item, "key", None)
        value = getattr(item, "value", None)
        if key is None:
            try:
                key, value = item
            except (TypeError, ValueError):
                continue
        if str(key).lower() == "trace_id":
            value = str(value or "").strip()
            if value:
                return value
    return str(uuid4())


def _chunk_answer(answer: str, target_chars: int):
    """Split a completed RAG answer into gRPC stream chunks without delays.

    The current RAG/LLM client is one-pass (non-streaming), therefore these chunks
    are emitted after generation. The mandatory FinalResult is sent afterwards.
    """
    buffer = ""
    for match in _TOKEN_SPLIT_RE.finditer(answer):
        piece = match.group(0)
        if buffer and len(buffer) + len(piece) > target_chars:
            yield buffer
            buffer = piece
        else:
            buffer += piece
    if buffer:
        yield buffer


def _check_llm(settings: Settings) -> tuple[bool, str]:
    url = f"{str(settings.llm_base_url).rstrip('/')}/models"
    headers = {"Authorization": f"Bearer {settings.llm_api_key}"}
    try:
        with httpx.Client(
            timeout=httpx.Timeout(
                settings.grpc_health_llm_timeout_seconds,
                connect=min(1.0, settings.grpc_health_llm_timeout_seconds),
            ),
            headers=headers,
        ) as client:
            response = client.get(url)
            if 200 <= response.status_code < 300:
                return True, "connected"
            return False, f"http_{response.status_code}"
    except httpx.HTTPError as exc:
        return False, f"unavailable:{type(exc).__name__}"


GRPC_OPTIONS = (
    ("grpc.max_receive_message_length", _MAX_GRPC_MESSAGE_BYTES),
    ("grpc.max_send_message_length", _MAX_GRPC_MESSAGE_BYTES),
)
