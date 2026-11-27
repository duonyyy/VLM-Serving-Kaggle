"""Shared multimodal chat and structured extraction execution helpers."""

from __future__ import annotations

import json
import time
import uuid
from typing import Any

from packages.contracts.chat import (
    ChatChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatResponseMessage,
    TimingBreakdown,
    UsageInfo,
)
from packages.contracts.extraction import ExtractionRequest, ExtractionResponse, InvoiceData
from packages.image_processing.decoder import decode_image
from packages.model_adapters.base import BaseModelAdapter


def parse_extraction_json(
    raw_text: str,
    schema_name: str | None = None,
) -> tuple[dict[str, Any], str]:
    """Parse JSON from model output with markdown-fence stripping and schema coercion."""
    stripped = raw_text.strip()
    if "```json" in stripped:
        candidate = stripped.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in stripped:
        candidate = stripped.split("```", 1)[1].split("```", 1)[0].strip()
    elif "{" in stripped and "}" in stripped:
        candidate = stripped[stripped.find("{") : stripped.rfind("}") + 1]
    else:
        return {"raw": raw_text}, "partial"

    try:
        parsed = json.loads(candidate)
    except Exception:
        return {"raw": raw_text}, "partial"

    if not isinstance(parsed, dict):
        return {"raw": raw_text}, "partial"

    if schema_name == "invoice":
        try:
            validated = InvoiceData.model_validate(parsed)
            merged = validated.model_dump()
            # Preserve any extra caller-requested keys while keeping validated core fields
            for key, val in parsed.items():
                if key not in merged:
                    merged[key] = val
            # Enforce evidence check: at least one core invoice field must be non-null/non-empty
            meaningful_fields = ["seller_name", "invoice_number", "invoice_date", "total_amount", "subtotal", "tax_amount", "items"]
            has_data = any(merged.get(f) is not None and merged.get(f) != [] for f in meaningful_fields)
            if not has_data:
                return merged, "partial"
            return merged, "completed"
        except Exception:
            return parsed, "partial"

    # For general dict schemas, if empty dict {}, mark partial
    if not parsed:
        return parsed, "partial"

    return parsed, "completed"


def build_extraction_prompt(payload: ExtractionRequest) -> str:
    """Build deterministic extraction prompt respecting explicit fields and schema."""
    if payload.prompt:
        return payload.prompt
    if payload.fields:
        fields_str = ", ".join(payload.fields)
        return f"Trích xuất các trường {fields_str} sang JSON."
    if payload.schema_name == "invoice":
        return (
            "Trích xuất thông tin hóa đơn sang JSON với các trường: "
            "seller_name, invoice_number, invoice_date, total_amount, currency."
        )
    return f"Trích xuất thông tin theo schema '{payload.schema_name or 'general'}' sang JSON."


def execute_chat_completion(
    adapter: BaseModelAdapter,
    payload: ChatCompletionRequest,
) -> ChatCompletionResponse:
    """Synchronously execute a ChatCompletionRequest on a loaded BaseModelAdapter."""
    start_e2e = time.perf_counter()
    user_prompt = payload.get_user_prompt()
    if not user_prompt:
        raise ValueError("Chat messages must contain at least one non-empty user prompt")

    image_urls = payload.get_image_urls()
    decoded_images = []
    prep_start = time.perf_counter()
    if image_urls:
        for img_src in image_urls:
            try:
                decoded_img, _ = decode_image(img_src)
                decoded_images.append(decoded_img)
            except Exception as exc:
                raise ValueError(f"Failed to process input image: {exc}") from exc
    image_prep_time_ms = (time.perf_counter() - prep_start) * 1000

    primary_image = decoded_images[0] if decoded_images else None
    infer_kwargs = dict(payload.extra_params)
    if len(decoded_images) > 1:
        infer_kwargs["images"] = decoded_images[1:]

    result = adapter.infer(
        prompt=user_prompt,
        image=primary_image,
        max_tokens=payload.max_tokens,
        temperature=payload.temperature,
        **infer_kwargs,
    )

    total_elapsed_ms = round((time.perf_counter() - start_e2e) * 1000, 3)
    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
        model=adapter.model_id,
        choices=[
            ChatChoice(
                index=0,
                message=ChatResponseMessage(content=result.prediction_text),
                finish_reason=result.finish_reason,
            )
        ],
        usage=UsageInfo(
            prompt_tokens=result.input_tokens,
            completion_tokens=result.output_tokens,
            total_tokens=result.input_tokens + result.output_tokens,
        ),
        timing=TimingBreakdown(
            preprocessing_ms=round(result.preprocessing_ms + image_prep_time_ms, 3),
            inference_ms=round(result.inference_ms, 3),
            total_ms=total_elapsed_ms,
        ),
    )


def execute_extraction(
    adapter: BaseModelAdapter,
    payload: ExtractionRequest,
    request_id: str,
) -> ExtractionResponse:
    """Synchronously execute an ExtractionRequest on a loaded BaseModelAdapter."""
    start_e2e = time.perf_counter()
    prep_start = time.perf_counter()
    try:
        decoded_image, _ = decode_image(payload.image_url)
    except Exception as exc:
        raise ValueError(f"Failed to process input image: {exc}") from exc
    image_prep_time_ms = (time.perf_counter() - prep_start) * 1000

    prompt = build_extraction_prompt(payload)
    result = adapter.infer(prompt=prompt, image=decoded_image, max_tokens=512)
    parsed_data, parse_status = parse_extraction_json(result.prediction_text, schema_name=payload.schema_name)
    total_elapsed_ms = round((time.perf_counter() - start_e2e) * 1000, 3)

    return ExtractionResponse(
        request_id=request_id,
        model=adapter.model_id,
        status=parse_status,
        data=parsed_data,
        raw_text=result.prediction_text,
        usage=UsageInfo(
            prompt_tokens=result.input_tokens,
            completion_tokens=result.output_tokens,
            total_tokens=result.input_tokens + result.output_tokens,
        ),
        timing=TimingBreakdown(
            preprocessing_ms=round(result.preprocessing_ms + image_prep_time_ms, 3),
            inference_ms=round(result.inference_ms, 3),
            total_ms=total_elapsed_ms,
        ),
    )
