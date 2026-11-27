"""FastAPI application for VLM Worker serving OpenAI-compatible API."""

from __future__ import annotations

import asyncio
import time
from typing import Any
from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from packages.contracts.chat import ChatCompletionRequest, ChatCompletionResponse
from packages.contracts.errors import ErrorDetail, ErrorResponse
from packages.contracts.extraction import ExtractionRequest, ExtractionResponse
from packages.contracts.models import HealthResponse, ModelCard, ModelListResponse, matches_model_id
from packages.telemetry.context import get_request_id, set_request_id
from packages.telemetry.logging import get_logger
from .executor import execute_chat_completion, execute_extraction
from .manager import ModelManager

logger = get_logger("worker_api")


def create_app(manager: ModelManager | None = None) -> FastAPI:
    """Factory creating and configuring the FastAPI worker application."""
    model_manager = manager or ModelManager()

    app = FastAPI(
        title="VLM Worker Runtime API",
        version="0.1.0",
        description="Low-latency OpenAI-compatible serving engine for Vision-Language Models",
    )

    app.state.manager = model_manager

    # 1. Request Tracing & Timing Middleware
    @app.middleware("http")
    async def request_tracing_middleware(request: Request, call_next: Any) -> Response:
        start_time = time.perf_counter()
        req_id_header = request.headers.get("X-Request-ID")
        req_id = set_request_id(req_id_header)

        try:
            response = await call_next(request)
        except Exception as exc:
            # Let custom exception handlers format the error response
            raise exc

        duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
        response.headers["X-Request-ID"] = req_id
        response.headers["X-Process-Time-Ms"] = str(duration_ms)

        logger.info(
            f"{request.method} {request.url.path} -> {response.status_code} ({duration_ms}ms)",
            extra={"extra_data": {"method": request.method, "path": request.url.path, "status_code": response.status_code, "duration_ms": duration_ms}},
        )
        return response

    # 2. Structured Error Handlers
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        req_id = get_request_id()
        error_msg = "; ".join(f"{'.'.join(str(loc) for loc in err['loc'])}: {err['msg']}" for err in exc.errors())
        logger.warning(f"Validation error: {error_msg}")
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=ErrorResponse(
                error=ErrorDetail(message=error_msg, type="validation_error", code="invalid_payload"),
                request_id=req_id,
            ).model_dump(),
            headers={"X-Request-ID": req_id},
        )

    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
        req_id = get_request_id()
        logger.warning(f"Bad request value error: {exc}")
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=ErrorResponse(
                error=ErrorDetail(message=str(exc), type="invalid_request_error", code="bad_input"),
                request_id=req_id,
            ).model_dump(),
            headers={"X-Request-ID": req_id},
        )

    @app.exception_handler(RuntimeError)
    async def runtime_error_handler(request: Request, exc: RuntimeError) -> JSONResponse:
        req_id = get_request_id()
        err_str = str(exc)
        is_unavailable = any(k in err_str.lower() for k in ("not ready", "not registered", "no model adapter", "model unavailable"))
        status_code = status.HTTP_503_SERVICE_UNAVAILABLE if is_unavailable else status.HTTP_500_INTERNAL_SERVER_ERROR
        logger.error(f"Runtime error ({status_code}): {exc}")
        return JSONResponse(
            status_code=status_code,
            content=ErrorResponse(
                error=ErrorDetail(message=err_str, type="runtime_error", code="model_unavailable" if is_unavailable else "internal_failure"),
                request_id=req_id,
            ).model_dump(),
            headers={"X-Request-ID": req_id},
        )

    # 3. Routes
    @app.get("/health", response_model=HealthResponse)
    async def health_endpoint() -> HealthResponse:
        return model_manager.get_health()

    @app.get("/v1/models", response_model=ModelListResponse)
    async def list_models_endpoint() -> ModelListResponse:
        cards = model_manager.list_models()
        return ModelListResponse(data=cards)

    @app.get("/v1/models/{model_id}", response_model=ModelCard)
    async def get_model_endpoint(model_id: str) -> ModelCard:
        cards = model_manager.list_models()
        for card in cards:
            if card.id == model_id:
                return card
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{model_id}' not found on this worker",
        )

    @app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
    async def chat_completions_endpoint(payload: ChatCompletionRequest) -> ChatCompletionResponse:
        adapter = model_manager.get_adapter()
        if payload.model and not matches_model_id(payload.model, adapter.model_id):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model '{payload.model}' not found on this worker (serving '{adapter.model_id}')",
            )
        async with model_manager.lock:
            return await asyncio.to_thread(execute_chat_completion, adapter, payload)

    @app.post("/v1/extract", response_model=ExtractionResponse)
    async def structured_extract_endpoint(payload: ExtractionRequest) -> ExtractionResponse:
        req_id = get_request_id()
        adapter = model_manager.get_adapter()
        if payload.model and not matches_model_id(payload.model, adapter.model_id):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model '{payload.model}' not found on this worker (serving '{adapter.model_id}')",
            )
        async with model_manager.lock:
            return await asyncio.to_thread(execute_extraction, adapter, payload, req_id)

    return app
