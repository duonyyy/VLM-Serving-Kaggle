"""Model metadata and worker health contracts."""

from __future__ import annotations

import time
from typing import Any, Literal
from pydantic import BaseModel, Field


class ModelCapabilities(BaseModel):
    image_chat: bool = True
    ocr_markdown: bool = False
    structured_json: bool = False
    max_image_edge: int = 1280
    supports_concurrency: bool = False
    max_concurrency: int = 1


class ModelCard(BaseModel):
    id: str
    object: Literal["model"] = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "vlm-platform"
    status: Literal["READY", "LOADING", "NOT_LOADED", "ERROR"] = "READY"
    capabilities: ModelCapabilities = Field(default_factory=ModelCapabilities)
    revision: str | None = None
    device: str | None = None


class ModelListResponse(BaseModel):
    object: Literal["list"] = "list"
    data: list[ModelCard]


class HealthResponse(BaseModel):
    status: Literal["healthy", "degraded", "unhealthy"]
    worker_id: str
    uptime_seconds: float
    active_model: str | None = None
    active_model_status: str = "NOT_LOADED"
    cuda_available: bool = False
    gpu: dict[str, Any] | None = None
    timestamp: str


class GatewayHealthResponse(BaseModel):
    status: Literal["healthy", "degraded", "unhealthy"]
    active_workers: int
    pending_jobs: int
    timestamp: float


MODEL_ALIASES: dict[str, list[str]] = {
    "Qwen/Qwen3-VL-2B-Instruct": [
        "qwen3-vl-2b",
        "qwen-3-vl-2b",
        "qwen3-vl-2b-instruct",
        "qwen/qwen3-vl-2b",
        "qwen/qwen3-vl-2b-instruct",
        "qwen3",
    ],
    "Qwen/Qwen2-VL-2B-Instruct": [
        "qwen2-vl-2b",
        "qwen-2-vl-2b",
        "qwen2-vl-2b-instruct",
        "qwen/qwen2-vl-2b",
        "qwen/qwen2-vl-2b-instruct",
        "qwen2",
    ],
    "mock-vlm-2b": [
        "mock",
        "mock-vlm",
        "mock-vlm-2b",
        "mock-remote-vlm",
    ],
}


def matches_model_id(requested: str | None, candidate: str) -> bool:
    """Check if requested model identifier matches candidate model or its aliases."""
    if not requested or requested == "any" or requested == "default":
        return True
    req = requested.strip().lower()
    cand = candidate.strip().lower()
    if req == cand:
        return True
    if req in cand or cand in req:
        return True
    for canonical, aliases in MODEL_ALIASES.items():
        canonical_lower = canonical.lower()
        alias_lowers = [a.lower() for a in aliases]
        all_group = [canonical_lower] + alias_lowers
        if (req in all_group) and (cand in all_group):
            return True
    return False
