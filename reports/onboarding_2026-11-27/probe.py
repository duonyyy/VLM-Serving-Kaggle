"""Read-only local diagnostics; writes audit evidence beside this script only.

Run from repository root: python reports/onboarding_2026-11-27/probe.py
Uses mock inference and in-process ASGI. Does not download weights or deploy.
"""
from __future__ import annotations

import ast
import asyncio
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient
import websockets
from apps.gateway.app import create_gateway_app
from apps.gateway.config import GatewayConfig
from apps.gateway.dispatcher import JobDispatcher
from apps.gateway.registry import WorkerRegistry
from packages.caching.cache import InferenceCache
from packages.contracts.protocol import WorkerJobResult, WorkerRegister
from packages.worker_runtime.app import create_app
from packages.worker_runtime.manager import ModelManager
from packages.worker_runtime.executor import parse_extraction_json
from packages.model_adapters.mock_adapter import MockAdapter
from client.vlm_client import VLMClient


def collect() -> dict:
    evidence = {"timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "scope": "CPU/mock/ASGI diagnostics; not GPU performance"}
    dispatcher = JobDispatcher(WorkerRegistry("audit-only"), cache=InferenceCache())

    def key(messages, extra=None):
        return dispatcher._extract_cache_key("/v1/chat/completions", {
            "model": "mock", "messages": messages, "extra_params": extra or {}
        }, "mock")

    evidence["cache_message_boundary_collision"] = key([
        {"role": "user", "content": "a b"}
    ]) == key([{"role": "user", "content": "a"}, {"role": "user", "content": "b"}])
    messages = [{"role": "user", "content": "hello"}]
    evidence["cache_ignores_extra_params"] = key(messages, {"top_p": .2}) == key(messages, {"top_p": .9})
    evidence["empty_invoice_parse"] = parse_extraction_json("{}", "invoice")
    manager = ModelManager()
    manager.set_adapter(MockAdapter(), {"simulate_latency_ms": 0})
    with TestClient(create_app(manager)) as client:
        response = client.post("/v1/chat/completions", json={
            "model": "does-not-exist", "messages": [{"role": "user", "content": "hi"}], "stream": True
        })
        evidence["worker_stream_wrong_model"] = {
            "status": response.status_code, "content_type": response.headers["content-type"],
            "returned_model": response.json().get("model")
        }
    with TestClient(create_gateway_app(GatewayConfig(worker_auth_token="audit-only"))) as client:
        try:
            VLMClient(http_client=client).get_health()
        except Exception as exc:
            evidence["sdk_gateway_health_error"] = type(exc).__name__ + ": " + str(exc).splitlines()[0]

    async def result_probe():
        dispatcher._pending_jobs["job-a"] = asyncio.get_running_loop().create_future()
        dispatcher._job_worker_map["job-a"] = "owner-session"
        evidence["result_accepts_without_session_check"] = dispatcher.resolve_job(WorkerJobResult(
            job_id="job-a", request_id="unrelated", status="completed", result={"synthetic": True}
        ))
        dispatcher._pending_jobs.clear()
        dispatcher._job_worker_map.clear()

    asyncio.run(result_probe())
    evidence["websocket_connect_default_max_size"] = inspect.signature(websockets.connect).parameters["max_size"].default
    registry = WorkerRegistry("audit-only")
    registry.register(WorkerRegister(worker_id="w", model_id="Qwen/Qwen3-VL-2B-Instruct", token="audit-only"), None)
    evidence["documented_short_alias_matches"] = registry.get_available_worker("qwen3-vl-2b") is not None

    from apps.gateway import server
    # Exercise the actual CLI config construction without starting a server.
    captured = {}
    def capture(config):
        captured["client_auth_enabled"] = bool(config.client_api_key)
        captured["worker_uses_environment_token"] = config.worker_auth_token == "audit-env-worker"
        return object()
    with patch.dict(os.environ, {"VLM_WORKER_AUTH_TOKEN": "audit-env-worker", "VLM_CLIENT_API_KEY": "audit-env-client"}), \
         patch.object(sys, "argv", ["gateway"]), \
         patch.object(server, "create_gateway_app", side_effect=capture), \
         patch.object(server.uvicorn, "run"):
        server.main()
    evidence["cli_environment_probe"] = captured
    evidence["versions"] = {p: importlib.metadata.version(p) for p in [
        "torch", "transformers", "fastapi", "pydantic", "websockets", "httpx", "pytest"
    ]}
    import torch
    evidence["torch_cuda"] = {"available": torch.cuda.is_available(), "runtime": torch.version.cuda,
                              "device_count": torch.cuda.device_count()}
    result = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                            capture_output=True, text=True)
    evidence["physical_gpu"] = result.stdout.strip()
    # Some managed hosts expose GIT_CONFIG_COUNT without its indexed values to
    # Python children. Omit command-scoped Git config for these read-only calls.
    git_env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_CONFIG_")}
    evidence["git_head"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, env=git_env).strip()
    evidence["branch"] = subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True, env=git_env).strip()
    return evidence


if __name__ == "__main__":
    outdir = Path(__file__).resolve().parent
    evidence = collect()
    inventory = []
    for directory in ["apps", "packages", "client", "evaluation", "tests", "research/ocr_benchmark"]:
        for path in sorted((ROOT / directory).rglob("*.py")):
            raw = path.read_bytes()
            tree = ast.parse(raw.decode("utf-8-sig"))
            inventory.append({"path": path.relative_to(ROOT).as_posix(), "sha256": hashlib.sha256(raw).hexdigest(),
                              "lines": len(raw.splitlines()), "test_methods": sum(
                                  isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_")
                                  for node in ast.walk(tree))})
    evidence["inventory_python_files"] = len(inventory)
    evidence["inventory_test_methods"] = sum(item["test_methods"] for item in inventory)
    (outdir / "source_inventory.json").write_text(json.dumps(inventory, indent=2) + "\n", encoding="utf-8")
    (outdir / "probe_results.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(evidence, ensure_ascii=True, indent=2))
