"""End-to-End integration test: Client -> Gateway -> Outbound WebSocket -> Worker -> Client."""

import asyncio
import json
import socket
import threading
import time
import unittest
import httpx
import uvicorn
import websockets

from apps.gateway.app import create_gateway_app
from apps.gateway.config import GatewayConfig
from packages.contracts.chat import ChatChoice, ChatCompletionResponse, ChatResponseMessage, TimingBreakdown, UsageInfo
from packages.contracts.extraction import ExtractionResponse
from packages.contracts.protocol import WorkerJobResult, WorkerRegister
from packages.model_adapters.mock_adapter import MockAdapter
from packages.worker_runtime.remote_client import RemoteWorkerClient


def get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


class TestGatewayWorkerE2E(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.port = get_free_port()
        cls.auth_token = "e2e_secret_token"
        cls.config = GatewayConfig(
            worker_auth_token=cls.auth_token,
            job_timeout_sec=5.0,
            heartbeat_timeout_sec=10.0,
        )
        cls.app = create_gateway_app(cls.config)
        server_config = uvicorn.Config(cls.app, host="127.0.0.1", port=cls.port, log_level="warning")
        cls.server = uvicorn.Server(server_config)
        cls.server_thread = threading.Thread(target=cls.server.run, daemon=True)
        cls.server_thread.start()
        time.sleep(0.4)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.should_exit = True
        cls.server_thread.join(timeout=2.0)

    def test_end_to_end_chat_completion(self) -> None:
        """Verify chat completion routed across real WebSocket to remote worker."""
        worker_ready = threading.Event()
        stop_worker = threading.Event()

        def worker_loop() -> None:
            async def run():
                ws_url = f"ws://127.0.0.1:{self.port}/workers/connect"
                async with websockets.connect(ws_url) as ws:
                    reg = WorkerRegister(
                        worker_id="remote_kaggle_t4",
                        model_id="Qwen/Qwen2-VL-2B-Instruct",
                        token=self.auth_token,
                    )
                    await ws.send(reg.model_dump_json())
                    ack_data = json.loads(await ws.recv())
                    self.assertEqual(ack_data["status"], "registered")
                    worker_ready.set()

                    while not stop_worker.is_set():
                        raw_msg = await ws.recv()
                        msg = json.loads(raw_msg)
                        if msg.get("type") == "job_dispatch":
                            resp_content = ChatCompletionResponse(
                                id=msg["request_id"],
                                model="Qwen/Qwen2-VL-2B-Instruct",
                                choices=[
                                    ChatChoice(
                                        index=0,
                                        message=ChatResponseMessage(
                                            role="assistant",
                                            content="Hóa đơn GTGT số HD-00123 đã được xử lý thành công.",
                                        ),
                                    )
                                ],
                                usage=UsageInfo(prompt_tokens=15, completion_tokens=20, total_tokens=35),
                                timing=TimingBreakdown(total_ms=45.0),
                            )
                            result_msg = WorkerJobResult(
                                job_id=msg["job_id"],
                                request_id=msg["request_id"],
                                status="completed",
                                result=resp_content.model_dump(),
                                timing=resp_content.timing,
                            )
                            await ws.send(result_msg.model_dump_json())
                            break

            asyncio.run(run())

        t = threading.Thread(target=worker_loop, daemon=True)
        t.start()
        self.assertTrue(worker_ready.wait(timeout=3.0))

        client = httpx.Client(base_url=f"http://127.0.0.1:{self.port}")
        post_resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "Qwen/Qwen2-VL-2B-Instruct",
                "messages": [{"role": "user", "content": "Phân tích hóa đơn này"}],
            },
        )
        stop_worker.set()
        t.join(timeout=2.0)

        self.assertEqual(post_resp.status_code, 200)
        data = post_resp.json()
        self.assertEqual(data["model"], "Qwen/Qwen2-VL-2B-Instruct")
        self.assertEqual(
            data["choices"][0]["message"]["content"],
            "Hóa đơn GTGT số HD-00123 đã được xử lý thành công.",
        )
        self.assertIn("timing", data)
        self.assertEqual(data["timing"]["total_ms"], 45.0)

    def test_end_to_end_extraction(self) -> None:
        """Verify extraction endpoint routed across real WebSocket to remote worker."""
        worker_ready = threading.Event()
        stop_worker = threading.Event()

        def worker_loop() -> None:
            async def run():
                ws_url = f"ws://127.0.0.1:{self.port}/workers/connect"
                async with websockets.connect(ws_url) as ws:
                    reg = WorkerRegister(
                        worker_id="remote_extractor_01",
                        model_id="Qwen/Qwen2-VL-2B-Instruct",
                        token=self.auth_token,
                    )
                    await ws.send(reg.model_dump_json())
                    await ws.recv()
                    worker_ready.set()

                    raw_msg = await ws.recv()
                    msg = json.loads(raw_msg)
                    if msg.get("type") == "job_dispatch":
                        extract_resp = ExtractionResponse(
                            request_id=msg["request_id"],
                            model="Qwen/Qwen2-VL-2B-Instruct",
                            data={
                                "invoice_number": "HD-999",
                                "total_amount": 1500000.0,
                            },
                            timing=TimingBreakdown(total_ms=50.0),
                        )
                        result_msg = WorkerJobResult(
                            job_id=msg["job_id"],
                            request_id=msg["request_id"],
                            status="completed",
                            result=extract_resp.model_dump(),
                            timing=extract_resp.timing,
                        )
                        await ws.send(result_msg.model_dump_json())

            asyncio.run(run())

        t = threading.Thread(target=worker_loop, daemon=True)
        t.start()
        self.assertTrue(worker_ready.wait(timeout=3.0))

        client = httpx.Client(base_url=f"http://127.0.0.1:{self.port}")
        post_resp = client.post(
            "/v1/extract",
            json={
                "model": "Qwen/Qwen2-VL-2B-Instruct",
                "image_url": "data:image/jpeg;base64,/9j/4AAQSkZJRg==",
                "schema_name": "invoice",
            },
        )
        stop_worker.set()
        t.join(timeout=2.0)

        self.assertEqual(post_resp.status_code, 200)
        data = post_resp.json()
        self.assertEqual(data["data"]["invoice_number"], "HD-999")
        self.assertEqual(data["data"]["total_amount"], 1500000.0)

    def test_real_remote_worker_client_e2e(self) -> None:
        """Verify real RemoteWorkerClient + MockAdapter end-to-end through Gateway."""
        import base64
        import io
        from PIL import Image

        adapter = MockAdapter(model_id="mock-remote-vlm")
        adapter.load({"simulate_latency_ms": 5.0})
        remote_client = RemoteWorkerClient(
            gateway_ws_url=f"ws://127.0.0.1:{self.port}/workers/connect",
            adapter=adapter,
            auth_token=self.auth_token,
            worker_id="real_remote_worker_01",
        )

        def run_remote() -> None:
            asyncio.run(remote_client.run())

        wt = threading.Thread(target=run_remote, daemon=True)
        wt.start()

        http_client = httpx.Client(base_url=f"http://127.0.0.1:{self.port}")
        # Wait until worker registers and readiness turns 200
        ready = False
        for _ in range(40):
            r = http_client.get("/health/ready")
            if r.status_code == 200:
                ready = True
                break
            time.sleep(0.05)
        self.assertTrue(ready, "RemoteWorkerClient failed to register with Gateway")

        try:
            img = Image.new("RGB", (64, 48), color="navy")
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            img_uri = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")

            chat_resp = http_client.post(
                "/v1/chat/completions",
                json={
                    "model": "mock-remote-vlm",
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "Mô tả ảnh này"},
                                {"type": "image_url", "image_url": {"url": img_uri}},
                            ],
                        }
                    ],
                },
            )
            self.assertEqual(chat_resp.status_code, 200, chat_resp.text)
            self.assertIn("64x48", chat_resp.json()["choices"][0]["message"]["content"])

            # Second identical request should hit Gateway cache and record cache metrics
            chat_resp_cached = http_client.post(
                "/v1/chat/completions",
                json={
                    "model": "mock-remote-vlm",
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "Mô tả ảnh này"},
                                {"type": "image_url", "image_url": {"url": img_uri}},
                            ],
                        }
                    ],
                },
            )
            self.assertEqual(chat_resp_cached.status_code, 200)
            self.assertTrue(chat_resp_cached.json().get("cached"))

            extract_resp = http_client.post(
                "/v1/extract",
                json={
                    "model": "mock-remote-vlm",
                    "image_url": img_uri,
                    "schema_name": "invoice",
                },
            )
            self.assertEqual(extract_resp.status_code, 200, extract_resp.text)
            extract_data = extract_resp.json()
            self.assertEqual(extract_data["status"], "completed")
            self.assertEqual(extract_data["data"]["seller_name"], "Công ty TNHH Cà Phê Việt")

            metrics_text = http_client.get("/metrics").text
            self.assertIn("vlm_cache_hits_total", metrics_text)
        finally:
            remote_client.stop()
            wt.join(timeout=2.0)


if __name__ == "__main__":
    unittest.main()
