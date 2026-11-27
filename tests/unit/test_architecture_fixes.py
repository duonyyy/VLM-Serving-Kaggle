"""Unit tests verifying fixes for ARCHITECTURE_REVIEW_2026-11-27 findings."""

from __future__ import annotations

import asyncio
import os
import sys
import unittest
from unittest.mock import patch
from fastapi.testclient import TestClient

from apps.gateway import server
from apps.gateway.app import create_gateway_app
from apps.gateway.config import GatewayConfig
from apps.gateway.dispatcher import JobDispatcher
from apps.gateway.registry import WorkerRegistry
from apps.gateway.scheduler import SchedulingStrategy, WorkerScheduler
from client.vlm_client import VLMClient
from evaluation.ablation_runner import AblationRunner
from evaluation.runner import EvaluationRunner
from packages.caching.cache import InferenceCache
from packages.contracts.models import GatewayHealthResponse, HealthResponse, matches_model_id
from packages.contracts.protocol import WorkerJobResult, WorkerRegister
from packages.model_adapters.mock_adapter import MockAdapter
from packages.worker_runtime.app import create_app
from packages.worker_runtime.executor import parse_extraction_json
from packages.worker_runtime.manager import ModelManager


class ArchitectureFixesTests(unittest.TestCase):
    """Test suite verifying all P0 and P1 architecture fixes."""

    def test_cli_env_precedence(self) -> None:
        """Verify environment variables take precedence when CLI arguments are omitted."""
        captured_config: list[GatewayConfig] = []

        def capture(config: GatewayConfig) -> object:
            captured_config.append(config)
            return object()

        with patch.dict(
            os.environ,
            {
                "VLM_WORKER_AUTH_TOKEN": "audit-env-worker-123",
                "VLM_CLIENT_API_KEY": "audit-env-client-456",
            },
        ), patch.object(sys, "argv", ["gateway"]), patch.object(
            server, "create_gateway_app", side_effect=capture
        ), patch.object(server.uvicorn, "run"):
            server.main()

        self.assertEqual(len(captured_config), 1)
        cfg = captured_config[0]
        self.assertEqual(cfg.worker_auth_token, "audit-env-worker-123")
        self.assertEqual(cfg.client_api_key, "audit-env-client-456")

    def test_sender_session_ownership_enforcement(self) -> None:
        """Verify results from mismatched sender sessions or request IDs are rejected."""
        dispatcher = JobDispatcher(WorkerRegistry("audit-token"), cache=InferenceCache())
        loop = asyncio.new_event_loop()
        future = loop.create_future()

        dispatcher._pending_jobs["job-test-1"] = future
        dispatcher._job_worker_map["job-test-1"] = "valid-session"
        dispatcher._job_request_map["job-test-1"] = "req-match"

        # 1. Reject mismatched sender session
        rejected_session = dispatcher.resolve_job(
            WorkerJobResult(
                job_id="job-test-1",
                request_id="req-match",
                status="completed",
                result={"ok": True},
            ),
            sender_session_id="foreign-session",
        )
        self.assertFalse(rejected_session)
        self.assertFalse(future.done())

        # 2. Reject mismatched request_id
        rejected_req = dispatcher.resolve_job(
            WorkerJobResult(
                job_id="job-test-1",
                request_id="wrong-req",
                status="completed",
                result={"ok": True},
            ),
            sender_session_id="valid-session",
        )
        self.assertFalse(rejected_req)
        self.assertFalse(future.done())

        # 3. Accept matching session and request_id
        accepted = dispatcher.resolve_job(
            WorkerJobResult(
                job_id="job-test-1",
                request_id="req-match",
                status="completed",
                result={"ok": True},
            ),
            sender_session_id="valid-session",
        )
        self.assertTrue(accepted)
        self.assertTrue(future.done())
        self.assertEqual(future.result().status, "completed")
        loop.close()

    def test_model_alias_matching(self) -> None:
        """Verify model alias normalization and matching for Qwen and Mock."""
        # Short alias to canonical
        self.assertTrue(matches_model_id("qwen3-vl-2b", "Qwen/Qwen3-VL-2B-Instruct"))
        self.assertTrue(matches_model_id("qwen2-vl-2b", "Qwen/Qwen2-VL-2B-Instruct"))
        self.assertTrue(matches_model_id("mock", "mock-vlm-2b"))
        self.assertTrue(matches_model_id("mock-remote-vlm", "mock-vlm-2b"))

        # Incompatible model
        self.assertFalse(matches_model_id("non-existent-model", "mock-vlm-2b"))
        self.assertFalse(matches_model_id("qwen3-vl-2b", "Qwen/Qwen2-VL-2B-Instruct"))

    def test_worker_runtime_wrong_model_returns_404(self) -> None:
        """Verify direct worker returns 404 when requested model is not served."""
        manager = ModelManager()
        manager.set_adapter(MockAdapter(), {"simulate_latency_ms": 0})
        app = create_app(manager)

        with TestClient(app) as client:
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": "unknown-model-xyz",
                    "messages": [{"role": "user", "content": "hello"}],
                },
            )
            self.assertEqual(resp.status_code, 404)
            self.assertIn("not found on this worker", resp.text)

    def test_cache_collision_prevention_and_fresh_envelope(self) -> None:
        """Verify message boundary differences and extra params produce distinct keys and fresh envelopes."""
        dispatcher = JobDispatcher(WorkerRegistry("audit-token"), cache=InferenceCache())

        def get_key(messages: list[dict], extra: dict | None = None) -> str | None:
            return dispatcher._extract_cache_key(
                "/v1/chat/completions",
                {
                    "model": "mock",
                    "messages": messages,
                    "extra_params": extra or {},
                },
                "mock",
            )

        # Boundary collision check
        k1 = get_key([{"role": "user", "content": "a b"}])
        k2 = get_key([{"role": "user", "content": "a"}, {"role": "user", "content": "b"}])
        self.assertIsNotNone(k1)
        self.assertIsNotNone(k2)
        self.assertNotEqual(k1, k2, "Message boundaries must not collide")

        # Extra params check
        k_p1 = get_key([{"role": "user", "content": "hello"}], {"top_p": 0.2})
        k_p2 = get_key([{"role": "user", "content": "hello"}], {"top_p": 0.9})
        self.assertNotEqual(k_p1, k_p2, "Different extra_params must produce distinct keys")

    def test_empty_invoice_parse_returns_partial(self) -> None:
        """Verify empty JSON extraction result returns 'partial' status."""
        data, status = parse_extraction_json("{}", "invoice")
        self.assertEqual(status, "partial")

        # When meaningful field exists, should be completed
        data_valid, status_valid = parse_extraction_json(
            '{"seller_name": "Công ty ABC", "total_amount": 500000}', "invoice"
        )
        self.assertEqual(status_valid, "completed")
        self.assertEqual(data_valid["seller_name"], "Công ty ABC")

    def test_sdk_gateway_health_compatibility(self) -> None:
        """Verify VLMClient.get_health() seamlessly parses Gateway health response."""
        gateway_app = create_gateway_app(GatewayConfig(worker_auth_token="test-tok"))
        with TestClient(gateway_app) as client:
            sdk = VLMClient(http_client=client)
            health = sdk.get_health()
            self.assertIsInstance(health, GatewayHealthResponse)
            self.assertIn(health.status, ("healthy", "degraded"))

    def test_evaluation_missing_dataset_fails(self) -> None:
        """Verify AblationRunner raises FileNotFoundError on missing dataset."""
        runner = AblationRunner(dataset_path="non_existent_dataset.jsonl")
        with self.assertRaises(FileNotFoundError):
            runner.load_dataset()


if __name__ == "__main__":
    unittest.main()
