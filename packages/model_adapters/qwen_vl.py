"""Qwen-VL Transformers Model Adapter for Kaggle T4 and Production Serving."""

from __future__ import annotations

import re
import time
from typing import Any
from PIL import Image

from packages.image_processing.dynamic_resizer import DynamicResizer, ResolutionPolicy
from packages.image_processing.resizer import prepare_image
from .base import BaseModelAdapter, InferenceResult

DEFAULT_MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
DEFAULT_MAX_EDGE = 1280


class QwenVLAdapter(BaseModelAdapter):
    """Production Model Adapter for Qwen3-VL and Qwen2-VL models."""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        max_image_edge: int = DEFAULT_MAX_EDGE,
        resolution_policy: ResolutionPolicy | None = None,
    ) -> None:
        super().__init__(model_id)
        self.max_image_edge = max_image_edge
        self.resolution_policy = resolution_policy
        self.resizer = DynamicResizer(resolution_policy) if resolution_policy else None
        self.revision: str | None = None
        self.device_name: str = "cuda:0"
        self.processor: Any = None
        self.model: Any = None
        self.torch: Any = None
        self._capabilities.image_chat = True
        self._capabilities.ocr_markdown = True
        self._capabilities.structured_json = True
        self._capabilities.max_image_edge = max_image_edge

    def load(self, config: dict[str, Any] | None = None) -> None:
        """Load Qwen-VL model weights with float16 and eager/sdpa attention."""
        self._status = "LOADING"
        cfg = config or {}
        try:
            # Validate revision before heavy torch/transformers imports
            req_rev = cfg.get("revision")
            if req_rev:
                if not re.fullmatch(r"[0-9a-fA-F]{40}", str(req_rev)):
                    raise ValueError("Model revision must be a 40-character commit SHA")
                self.revision = str(req_rev)

            import torch
            import transformers
            from transformers import AutoProcessor

            self.torch = torch
            target_device = cfg.get("device")
            if target_device:
                self.device_name = str(target_device)
            else:
                self.device_name = "cuda:0" if torch.cuda.is_available() else "cpu"

            # Determine precision & quantization
            dtype_req = cfg.get("torch_dtype") or cfg.get("dtype")
            if dtype_req == "bfloat16" and hasattr(torch, "bfloat16"):
                dtype = torch.bfloat16
            elif dtype_req == "float32":
                dtype = torch.float32
            else:
                dtype = torch.float16 if self.device_name.startswith("cuda") else torch.float32

            quant_mode = str(cfg.get("quantization", "none")).lower()

            processor_kwargs: dict[str, Any] = {}
            if self.revision:
                processor_kwargs["revision"] = self.revision
            if "min_pixels" in cfg:
                processor_kwargs["min_pixels"] = int(cfg["min_pixels"])
            if "max_pixels" in cfg:
                processor_kwargs["max_pixels"] = int(cfg["max_pixels"])

            # Load processor
            self.processor = AutoProcessor.from_pretrained(
                self.model_id,
                **processor_kwargs,
            )

            # Select appropriate model class for Qwen2-VL vs Qwen3-VL
            model_id_lower = self.model_id.lower()
            if "qwen2-vl" in model_id_lower or "qwen2.5-vl" in model_id_lower:
                model_cls = getattr(transformers, "Qwen2_5_VLForConditionalGeneration", None) if "qwen2.5" in model_id_lower else getattr(transformers, "Qwen2VLForConditionalGeneration", None)
                if model_cls is None:
                    model_cls = getattr(transformers, "AutoModelForVision2Seq")
            else:
                model_cls = getattr(transformers, "Qwen3VLForConditionalGeneration", None) or getattr(transformers, "AutoModelForVision2Seq")

            attn_impl = str(cfg.get("attn_implementation", "eager"))
            load_kwargs: dict[str, Any] = {
                "dtype": dtype,
                "attn_implementation": attn_impl,
            }
            if self.revision:
                load_kwargs["revision"] = self.revision

            if quant_mode in {"int8", "8bit"}:
                load_kwargs["load_in_8bit"] = True
            elif quant_mode in {"int4", "4bit"}:
                load_kwargs["load_in_4bit"] = True

            if self.device_name.startswith("cuda"):
                load_kwargs["device_map"] = self.device_name

            self.model = model_cls.from_pretrained(
                self.model_id,
                **load_kwargs,
            ).eval()

            if not self.device_name.startswith("cuda") and quant_mode == "none":
                self.model = self.model.to(self.device_name)

            self._status = "READY"
            self._error_message = None


        except Exception as exc:
            self._status = "ERROR"
            self._error_message = f"Failed to load Qwen model: {exc}"
            raise RuntimeError(self._error_message) from exc

    def infer(
        self,
        prompt: str,
        image: Image.Image | None = None,
        max_tokens: int = 256,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> InferenceResult:
        if self._status != "READY" or self.model is None or self.processor is None:
            raise RuntimeError(f"Qwen adapter is not ready (current status: {self._status})")

        torch = self.torch
        is_cuda = self.device_name.startswith("cuda")

        if is_cuda:
            torch.cuda.synchronize(0)
            torch.cuda.reset_peak_memory_stats(0)

        # Preprocessing phase
        prep_start = time.perf_counter()
        content_parts: list[dict[str, Any]] = []

        oriented_size = (0, 0)
        processed_size = (0, 0)
        extra_images: list[Image.Image] = list(kwargs.get("images") or [])
        input_images = ([image] if image is not None else []) + extra_images
        for idx_img, img in enumerate(input_images):
            target_edge = self.max_image_edge
            if self.resizer is not None:
                # Orient first, then apply DynamicResizer policy
                oriented_img, o_size, _ = prepare_image(img, max_edge=max(img.size) + 1)
                processed_img, res_meta = self.resizer.optimize_image(oriented_img)
                p_size = res_meta["optimized_size"]
            else:
                processed_img, o_size, p_size = prepare_image(img, target_edge)
            if idx_img == 0:
                oriented_size = o_size
                processed_size = p_size
            content_parts.append({"type": "image", "image": processed_img})

        content_parts.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content_parts}]

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)

        if is_cuda:
            torch.cuda.synchronize(0)
        preprocessing_ms = (time.perf_counter() - prep_start) * 1000

        # Generation phase
        infer_start = time.perf_counter()
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": max_tokens,
            "do_sample": temperature > 0.0,
        }
        if temperature > 0.0:
            gen_kwargs["temperature"] = temperature

        with torch.inference_mode():
            generated = self.model.generate(**inputs, **gen_kwargs)

        if is_cuda:
            torch.cuda.synchronize(0)
        inference_ms = (time.perf_counter() - infer_start) * 1000

        input_tokens = int(inputs["input_ids"].shape[-1])
        output_ids = generated[0][input_tokens:]
        output_tokens = int(output_ids.shape[-1])
        prediction_text = self.processor.decode(
            output_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()

        extra_info: dict[str, Any] = {
            "device": self.device_name,
            "oriented_size": oriented_size,
            "processed_size": processed_size,
        }
        if is_cuda:
            extra_info["peak_torch_allocated_bytes"] = int(torch.cuda.max_memory_allocated(0))
            extra_info["peak_torch_reserved_bytes"] = int(torch.cuda.max_memory_reserved(0))

        finish_reason = "length" if output_tokens >= max_tokens else "stop"

        return InferenceResult(
            prediction_text=prediction_text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            preprocessing_ms=round(preprocessing_ms, 3),
            inference_ms=round(inference_ms, 3),
            finish_reason=finish_reason,
            extra=extra_info,
        )

    def unload(self) -> None:
        self.model = None
        self.processor = None
        if self.torch and self.device_name.startswith("cuda"):
            self.torch.cuda.empty_cache()
        self._status = "NOT_LOADED"
        self._error_message = None
