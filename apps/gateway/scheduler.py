"""Advanced Worker Scheduling and Load Balancing for Gateway Control Plane."""

from __future__ import annotations

from enum import Enum
from typing import Sequence

from apps.gateway.registry import ConnectedWorker
from packages.contracts.models import matches_model_id
from packages.telemetry.logging import get_logger

logger = get_logger("gateway_scheduler")


class SchedulingStrategy(str, Enum):
    LEAST_LOADED = "least_loaded"
    ROUND_ROBIN = "round_robin"
    MODEL_AFFINITY = "model_affinity"


class WorkerScheduler:
    """Dispatches inference requests to the most optimal remote worker based on policy."""

    def __init__(self, strategy: SchedulingStrategy = SchedulingStrategy.LEAST_LOADED) -> None:
        self.strategy = strategy
        self._round_robin_index = 0

    def select_worker(
        self,
        workers: Sequence[ConnectedWorker],
        model_id: str | None = None,
    ) -> ConnectedWorker | None:
        """Select the best eligible worker according to active strategy.

        Only idle workers matching the requested model are considered.
        """
        if not workers:
            return None

        # Filter candidates matching requested model and available capacity
        candidates = [
            w for w in workers
            if (
                w.status == "idle"
                or (
                    w.status == "busy"
                    and w.capabilities.supports_concurrency
                    and w.in_flight_jobs < w.max_concurrency
                )
            )
            and (matches_model_id(model_id, w.model_id))
        ]

        if not candidates:
            return None

        if self.strategy == SchedulingStrategy.ROUND_ROBIN:
            return self._select_round_robin(candidates)
        elif self.strategy == SchedulingStrategy.MODEL_AFFINITY:
            return self._select_model_affinity(candidates, model_id)
        else:
            return self._select_least_loaded(candidates)

    def _select_least_loaded(self, candidates: list[ConnectedWorker]) -> ConnectedWorker:
        """Pick worker with lowest in-flight utilization, breaking ties by oldest heartbeat."""
        candidates.sort(
            key=lambda w: (w.in_flight_jobs / max(1, w.max_concurrency), w.in_flight_jobs, w.last_heartbeat)
        )
        chosen = candidates[0]
        logger.debug(f"[LeastLoaded] Selected worker '{chosen.worker_id}' (session: {chosen.session_id})")
        return chosen

    def _select_round_robin(self, candidates: list[ConnectedWorker]) -> ConnectedWorker:
        """Select worker in circular rotation."""
        self._round_robin_index = (self._round_robin_index + 1) % len(candidates)
        chosen = candidates[self._round_robin_index]
        logger.debug(f"[RoundRobin] Selected index {self._round_robin_index} -> worker '{chosen.worker_id}'")
        return chosen

    def _select_model_affinity(self, candidates: list[ConnectedWorker], model_id: str | None) -> ConnectedWorker:
        """Prioritize exact model string match before partial substring."""
        if model_id:
            exact_matches = [w for w in candidates if matches_model_id(model_id, w.model_id)]
            if exact_matches:
                return exact_matches[0]
        return candidates[0]
