"""Job Dispatcher and Queue Manager for Gateway Control Plane with Caching and Scheduling."""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import Any

from packages.caching.cache import InferenceCache
from packages.contracts.protocol import GatewayJobCancel, GatewayJobDispatch, WorkerJobResult
from packages.telemetry.logging import get_logger
from packages.telemetry.metrics import global_metrics
from .registry import ConnectedWorker, WorkerRegistry
from .scheduler import SchedulingStrategy, WorkerScheduler

logger = get_logger("gateway_dispatcher")


class NoWorkerAvailableError(Exception):
    """Raised when no remote worker is registered or capable of processing the request."""


class QueueFullError(Exception):
    """Raised when the gateway pending job queue exceeds capacity."""


class JobTimeoutError(Exception):
    """Raised when a worker fails to return a result within the allotted time."""


class RemoteWorkerExecutionError(Exception):
    """Raised when the remote worker returns a failed status with an error message."""


class JobDispatcher:
    """Manages job dispatching, queueing, caching, and result resolution over WebSockets."""

    def __init__(
        self,
        registry: WorkerRegistry,
        max_queue_size: int = 100,
        default_timeout_sec: float = 60.0,
        cache: InferenceCache | None = None,
        scheduler: WorkerScheduler | None = None,
    ) -> None:
        self.registry = registry
        self.max_queue_size = max_queue_size
        self.default_timeout_sec = default_timeout_sec
        self.cache = cache
        self.scheduler = scheduler or WorkerScheduler(SchedulingStrategy.LEAST_LOADED)

        # job_id -> asyncio.Future[WorkerJobResult]
        self._pending_jobs: dict[str, asyncio.Future[WorkerJobResult]] = {}
        # job_id -> session_id
        self._job_worker_map: dict[str, str] = {}
        # job_id -> request_id
        self._job_request_map: dict[str, str] = {}
        # Concurrency / Queue lock
        self._lock = asyncio.Lock()

    @property
    def pending_count(self) -> int:
        return len(self._pending_jobs)

    def _extract_cache_key(self, endpoint: str, payload: dict[str, Any], model_id: str | None) -> str | None:
        """Construct deterministic cache key preserving message boundaries, roles, and extra params."""
        if self.cache is None:
            return None

        # Non-deterministic sampling or streaming bypasses cache
        if payload.get("stream", False):
            return None
        if payload.get("temperature", 0.0) != 0.0:
            return None

        model = model_id or payload.get("model") or "unknown"
        extra: dict[str, Any] = {
            "temp": payload.get("temperature", 0.0),
            "max_tokens": payload.get("max_tokens"),
        }
        if payload.get("extra_params"):
            extra["extra_params"] = payload["extra_params"]

        img_parts: list[str] = []
        prompt_repr = ""

        if endpoint == "/v1/chat/completions":
            messages = payload.get("messages") or []
            # Preserve message boundaries and roles canonically
            canonical_messages = []
            for msg in messages:
                role = msg.get("role", "")
                content = msg.get("content", "")
                if isinstance(content, str):
                    canonical_messages.append({"role": role, "content": content})
                elif isinstance(content, list):
                    parts = []
                    for part in content:
                        if isinstance(part, dict):
                            if part.get("type") == "text":
                                parts.append({"type": "text", "text": part.get("text", "")})
                            elif part.get("type") == "image_url":
                                img_info = part.get("image_url", {})
                                url = img_info.get("url", "") if isinstance(img_info, dict) else ""
                                parts.append({"type": "image_url", "url": url})
                                if url:
                                    img_parts.append(url)
                    canonical_messages.append({"role": role, "content": parts})
            prompt_repr = json.dumps(canonical_messages, sort_keys=True, ensure_ascii=False)

        elif endpoint == "/v1/extract":
            prompt = payload.get("prompt") or payload.get("schema_name") or "extract"
            if payload.get("image_url"):
                img_parts.append(str(payload["image_url"]))
            extra["schema_name"] = payload.get("schema_name")
            extra["fields"] = payload.get("fields")
            prompt_repr = prompt

        if not img_parts and not prompt_repr:
            return None

        return InferenceCache.generate_key(
            endpoint=endpoint,
            model=model,
            prompt=prompt_repr,
            image_bytes_or_url="|".join(img_parts),
            extra_params=extra,
        )

    async def dispatch_job(
        self,
        endpoint: str,
        payload: dict[str, Any],
        request_id: str,
        model_id: str | None = None,
        timeout_sec: float | None = None,
        wait_for_worker_sec: float = 2.0,
    ) -> dict[str, Any]:
        """Dispatch a job to an available remote worker or return cached result."""
        # 1. Check Inference Cache
        cache_key = self._extract_cache_key(endpoint, payload, model_id)
        if cache_key and self.cache is not None:
            cached_result = self.cache.get(cache_key)
            global_metrics.record_cache(hit=cached_result is not None)
            if cached_result is not None:
                logger.info(f"Cache HIT for request '{request_id}' (key: {cache_key[:12]}...)")
                fresh_result = json.loads(json.dumps(cached_result))
                if endpoint == "/v1/chat/completions":
                    fresh_result["id"] = f"chatcmpl-{uuid.uuid4().hex[:12]}"
                    fresh_result["created"] = int(time.time())
                elif endpoint == "/v1/extract":
                    fresh_result["request_id"] = request_id

                fresh_result["timing"] = {
                    "preprocessing_ms": 0.0,
                    "inference_ms": 0.0,
                    "total_ms": 0.5,
                }
                fresh_result["cached"] = True
                return fresh_result

        timeout = timeout_sec or self.default_timeout_sec
        job_id = f"job_{uuid.uuid4().hex[:12]}"
        loop = asyncio.get_running_loop()
        future: asyncio.Future[WorkerJobResult] = loop.create_future()

        # Atomically check capacity and reserve queue slot before acquiring a worker
        async with self._lock:
            if len(self._pending_jobs) >= self.max_queue_size:
                logger.warning(f"Queue full ({len(self._pending_jobs)}/{self.max_queue_size}), rejecting {request_id}")
                raise QueueFullError("Gateway queue is full. Please try again later.")
            self._pending_jobs[job_id] = future

        worker: ConnectedWorker | None = None
        try:
            # 2. Acquire Worker using Scheduler policy
            worker = await self._acquire_worker(model_id=model_id, timeout_sec=wait_for_worker_sec)
            if not worker:
                logger.error(f"No worker available for model '{model_id}' (request: {request_id})")
                raise NoWorkerAvailableError(f"No available remote worker serving model '{model_id or 'any'}'")

            self._job_worker_map[job_id] = worker.session_id
            self._job_request_map[job_id] = request_id
            worker.in_flight_jobs += 1
            worker.status = "busy"
            worker.active_job_id = job_id

            dispatch_msg = GatewayJobDispatch(
                job_id=job_id,
                request_id=request_id,
                endpoint=endpoint,
                payload=payload,
                timeout_sec=timeout,
            )

            logger.info(
                f"Dispatching job '{job_id}' to worker '{worker.worker_id}' (session: {worker.session_id}, endpoint: {endpoint})"
            )

            # Send dispatch message over WebSocket
            await worker.websocket.send_text(dispatch_msg.model_dump_json())
            logger.info(f"Successfully sent dispatch to worker '{worker.worker_id}'")

            # Wait for result with timeout
            job_result = await asyncio.wait_for(future, timeout=timeout)

            if job_result.status == "failed":
                raise RemoteWorkerExecutionError(job_result.error or "Remote worker failed without details")

            res_data = job_result.result or {}

            # Populate cache if successful
            if cache_key and self.cache is not None:
                self.cache.set(cache_key, res_data)

            return res_data

        except asyncio.TimeoutError:
            logger.error(f"Job '{job_id}' timed out after {timeout}s on worker '{worker.worker_id if worker else 'unknown'}'")
            if worker is not None:
                try:
                    cancel_msg = GatewayJobCancel(job_id=job_id, reason="Gateway timeout exceeded")
                    await worker.websocket.send_text(cancel_msg.model_dump_json())
                except Exception:
                    pass
            raise JobTimeoutError(f"Worker did not respond within {timeout} seconds")

        finally:
            self._cleanup_job(job_id, worker)

    async def _acquire_worker(
        self,
        model_id: str | None = None,
        timeout_sec: float = 2.0,
    ) -> ConnectedWorker | None:
        """Attempt to get an eligible worker according to scheduling strategy."""
        start_time = time.time()
        while True:
            self.registry.prune_stale_workers()
            active_workers = list(self.registry.sessions.values())

            if self.scheduler:
                worker = self.scheduler.select_worker(active_workers, model_id=model_id)
            else:
                worker = self.registry.get_available_worker(model_id=model_id)

            if worker:
                return worker

            # If there are no workers at all registered, fail fast
            if self.registry.active_worker_count == 0:
                return None

            # If workers are registered but busy, wait up to timeout_sec
            if time.time() - start_time >= timeout_sec:
                return None

            await asyncio.sleep(0.05)

    def resolve_job(self, result: WorkerJobResult, sender_session_id: str | None = None) -> bool:
        """Resolve a pending job future when a WorkerJobResult arrives with session and request verification."""
        future = self._pending_jobs.get(result.job_id)
        if not future or future.done():
            logger.warning(f"Received result for unknown or already completed job: {result.job_id}")
            return False

        expected_session = self._job_worker_map.get(result.job_id)
        if expected_session and sender_session_id != expected_session:
            logger.warning(
                f"Rejecting result for job '{result.job_id}': sender session '{sender_session_id}' "
                f"does not match expected owner session '{expected_session}'"
            )
            return False

        expected_request = self._job_request_map.get(result.job_id)
        if expected_request and result.request_id and result.request_id != expected_request:
            logger.warning(
                f"Rejecting result for job '{result.job_id}': request_id '{result.request_id}' "
                f"does not match expected '{expected_request}'"
            )
            return False

        future.set_result(result)
        logger.info(f"Job '{result.job_id}' resolved successfully with status: {result.status}")
        return True

    def handle_worker_disconnect(self, session_id: str) -> None:
        """Fail all pending jobs assigned to a disconnected worker."""
        orphaned_jobs = [
            jid for jid, sid in self._job_worker_map.items() if sid == session_id
        ]
        for jid in orphaned_jobs:
            future = self._pending_jobs.get(jid)
            if future and not future.done():
                logger.warning(f"Aborting job '{jid}' due to worker disconnect (session: {session_id})")
                future.set_exception(
                    NoWorkerAvailableError("Remote worker disconnected during job execution")
                )
            self._job_worker_map.pop(jid, None)
            self._job_request_map.pop(jid, None)
            self._pending_jobs.pop(jid, None)

    def _cleanup_job(self, job_id: str, worker: ConnectedWorker | None) -> None:
        """Clean up pending state and release worker capacity."""
        self._pending_jobs.pop(job_id, None)
        self._job_worker_map.pop(job_id, None)
        self._job_request_map.pop(job_id, None)

        if worker is not None and worker.session_id in self.registry.sessions:
            worker.in_flight_jobs = max(0, worker.in_flight_jobs - 1)
            if worker.in_flight_jobs == 0:
                worker.status = "idle"
                worker.active_job_id = None
