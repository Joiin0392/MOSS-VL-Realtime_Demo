"""Standalone vision protocol tests (调用端到模型的实时视觉协议 §5).

Run:  <repo>/.venv/bin/python -m pytest server/tests/test_gateway_vision.py -q

Drives WS /v1/realtime end-to-end against the FakeSglangOmniServer through a
real SglangOmniPool + uvicorn rig (same harness shape as test_gateway_rest):
the full happy round (start → ready → batch frames → frame_acks → outputs with
control-token stripping + <|im_end|> end marker → stop), the busy retry
trigger ("realtime session is already active"), bad-image rejection, abrupt
disconnect cleanup with slot recovery, and the lenient unknown-field contract.
"""
from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List, Optional

import uvicorn
import websockets
from fastapi import FastAPI

from server.adapters.vlm.moss_vl_sglang_omni.pool import SglangOmniPool
from server.config import Settings
from server.gateway import vision as gateway_vision
from server.tests.test_sglang_omni_adapter import JPEG, FakeSglangOmniServer

START_PAYLOAD: Dict[str, Any] = {
    "type": "start",
    "prompt": "请描述这些画面中正在发生的事情。",
    "frame_queue_size": 32,
    "max_new_tokens": 512,
    "max_tokens_per_second": 160,
    "do_sample": False,
    "temperature": 0.2,
    "top_k": 20,
    "top_p": 0.8,
    "repetition_penalty": 1.05,
}


class Rig:
    def __init__(self, server: uvicorn.Server, task: asyncio.Task, host: str,
                 pool: SglangOmniPool):
        self.server = server
        self.task = task
        self.host = host
        self.base = f"http://{host}"
        self.ws_base = f"ws://{host}"
        self.pool = pool

    async def stop(self) -> None:
        self.server.should_exit = True
        await asyncio.wait_for(self.task, timeout=10)


async def start_vision(fake: FakeSglangOmniServer, pool: Optional[SglangOmniPool] = None,
                       **overrides: Any) -> Rig:
    kwargs = dict(
        sglang_omni_urls=fake.url,
        sglang_omni_connect_timeout_s=5.0,
        sglang_omni_health_interval_s=600.0,  # prober effectively off in tests
        vision_start_timeout_s=5.0,
        vision_frame_timeout_s=6.0,
        vision_round_timeout_s=30.0,
    )
    kwargs.update(overrides)
    settings = Settings(**kwargs)
    if pool is None:
        pool = SglangOmniPool(settings)
        pool.capacity_retry_delay_s = 0.05
    await asyncio.to_thread(pool.load, "", -1, "online_streaming")
    app = FastAPI(title="vision-plane-test")
    app.include_router(gateway_vision.router)
    app.state.vision_pool = pool

    config = uvicorn.Config(app, host="127.0.0.1", port=0, ws_max_size=64 * 1024 * 1024,
                            log_level="warning", lifespan="off")
    server = uvicorn.Server(config)
    task = asyncio.create_task(server.serve())
    while not server.started:
        await asyncio.sleep(0.02)
        assert not task.done(), "uvicorn failed to start"
    port = server.servers[0].sockets[0].getsockname()[1]
    return Rig(server, task, f"127.0.0.1:{port}", pool)


async def recv_json(ws: Any, timeout: float = 5.0) -> Dict[str, Any]:
    raw = await asyncio.wait_for(ws.recv(), timeout)
    assert isinstance(raw, str), f"expected a text message, got {type(raw)}"
    return json.loads(raw)


# ---------------------------------------------------------------- happy round


async def _send_start(ws: Any, **overrides: Any) -> None:
    payload = dict(START_PAYLOAD)
    payload.update(overrides)
    await ws.send(json.dumps(payload, ensure_ascii=False))


async def _upload_frame(ws: Any, timestamp: float, data: bytes = JPEG) -> None:
    await ws.send(json.dumps({"type": "frame", "timestamp": timestamp}))
    await ws.send(data)


async def _test_vision_round_happy_path() -> None:
    fake = FakeSglangOmniServer().start()
    rig = await start_vision(fake)
    try:
        # empty session_id= query param must be accepted (caller contract)
        async with websockets.connect(
                f"{rig.ws_base}/v1/realtime?session_id=") as ws:
            await _send_start(ws)
            ready = await recv_json(ws)
            assert ready == {"type": "ready"}, ready

            # the whole batch goes out BEFORE reading acks (§5.3 batch arrivals)
            for i in range(4):
                await _upload_frame(ws, timestamp=1.0 + i)
            acks = [await recv_json(ws) for _ in range(4)]
            assert acks == [{"type": "frame_ack"}] * 4, acks

            # script the model response once every binary reached the backend
            while len(fake.binaries) < 4:
                await asyncio.sleep(0.02)
            fake.send_delta("画面中有")
            fake.send_delta("一名行人正在通过路口。")
            fake.send_silence()

            # deltas cross stripped of control tokens; silence → the end marker
            outputs = [await recv_json(ws) for _ in range(3)]
            assert outputs == [
                {"type": "output", "text": "画面中有"},
                {"type": "output", "text": "一名行人正在通过路口。"},
                {"type": "output", "text": "<|im_end|>"},
            ], outputs

            # stop → no stop-ack on the wire, the server just tears down
            await ws.send(json.dumps({"type": "stop"}))

        # configure mapping at the omni backend
        cfg = fake.configure_payload
        assert cfg is not None and cfg["type"] == "session.configure"
        assert cfg["prompt"] == START_PAYLOAD["prompt"]
        assert cfg["max_new_tokens"] == 512
        assert cfg["max_tokens_per_turn"] == 160.0
        assert cfg["temperature"] == 0.0  # do_sample=false → greedy
        assert cfg["top_p"] == 0.8
        assert cfg["input_queue_capacity"] == 32  # from frame_queue_size
        assert set(cfg) <= {"type", "prompt", "system_prompt", "max_new_tokens",
                            "max_tokens_per_turn", "temperature", "top_p",
                            "input_queue_capacity"}  # extra=forbid-safe
        # frames crossed as raw JPEG, timestamps forwarded in order
        assert fake.binaries == [JPEG] * 4
        timestamps = [m["timestamp"] for m in fake.received if m.get("type") == "input.frame"]
        assert timestamps == [1.0, 2.0, 3.0, 4.0], timestamps
        # stop released the replica slot → the next round can start immediately
        await asyncio.sleep(0.2)
        assert rig.pool.busy == 0, rig.pool.status()
        print("happy round (ready/batch acks/outputs/end marker/stop): OK")
    finally:
        await rig.stop()
        fake.close()


def test_vision_round_happy_path() -> None:
    asyncio.run(_test_vision_round_happy_path())


# ---------------------------------------------------------------------- busy


async def _test_vision_busy_retry_trigger() -> None:
    fake = FakeSglangOmniServer(max_sessions=1).start()
    rig = await start_vision(fake)
    try:
        # round A occupies the only slot
        ws_a = await websockets.connect(f"{rig.ws_base}/v1/realtime")
        await _send_start(ws_a)
        assert (await recv_json(ws_a))["type"] == "ready"

        # round B → busy error carrying the caller's retry trigger string
        async with websockets.connect(f"{rig.ws_base}/v1/realtime") as ws_b:
            await _send_start(ws_b)
            err = await recv_json(ws_b)
            assert err["type"] == "error", err
            assert "realtime session is already active" in err["message"], err
        # no session leaked by the rejected round
        assert rig.pool.busy == 1

        # A stops → the slot frees → the next round succeeds
        await ws_a.send(json.dumps({"type": "stop"}))
        await asyncio.sleep(0.3)
        assert rig.pool.busy == 0
        async with websockets.connect(f"{rig.ws_base}/v1/realtime") as ws_c:
            await _send_start(ws_c)
            assert (await recv_json(ws_c))["type"] == "ready"
            await ws_c.send(json.dumps({"type": "stop"}))
        await asyncio.sleep(0.2)
        assert rig.pool.busy == 0
        print("busy → 'realtime session is already active' + recovery: OK")
    finally:
        await rig.stop()
        fake.close()


def test_vision_busy_retry_trigger() -> None:
    asyncio.run(_test_vision_busy_retry_trigger())


# ----------------------------------------------------------------- bad image


async def _test_vision_bad_image() -> None:
    fake = FakeSglangOmniServer().start()
    rig = await start_vision(fake)
    try:
        async with websockets.connect(f"{rig.ws_base}/v1/realtime") as ws:
            await _send_start(ws)
            assert (await recv_json(ws))["type"] == "ready"
            # not a JPEG/PNG/WebP magic → the round fails with a readable error
            await _upload_frame(ws, timestamp=0.0, data=b"GIF89a-not-an-image")
            err = await recv_json(ws)
            assert err["type"] == "error" and "frame rejected" in err["message"], err
        # the session was released; the very next round works
        await asyncio.sleep(0.2)
        assert rig.pool.busy == 0
        async with websockets.connect(f"{rig.ws_base}/v1/realtime") as ws:
            await _send_start(ws)
            assert (await recv_json(ws))["type"] == "ready"
            await _upload_frame(ws, timestamp=1.0)
            assert (await recv_json(ws)) == {"type": "frame_ack"}
            await ws.send(json.dumps({"type": "stop"}))
        print("bad image → error + slot recovery: OK")
    finally:
        await rig.stop()
        fake.close()


def test_vision_bad_image() -> None:
    asyncio.run(_test_vision_bad_image())


# ------------------------------------------------- disconnect cleanup (§5.5)


async def _test_vision_disconnect_cleanup() -> None:
    fake = FakeSglangOmniServer().start()
    rig = await start_vision(fake)
    try:
        ws = await websockets.connect(f"{rig.ws_base}/v1/realtime")
        await _send_start(ws)
        assert (await recv_json(ws))["type"] == "ready"
        await _upload_frame(ws, timestamp=1.0)
        assert (await recv_json(ws)) == {"type": "frame_ack"}
        # the caller vanishes mid-round without stop: the session must be
        # aborted and the slot freed for the next call
        await ws.close()
        for _ in range(60):
            if rig.pool.busy == 0 and fake.aborts >= 1:
                break
            await asyncio.sleep(0.05)
        assert rig.pool.busy == 0, rig.pool.status()
        assert fake.aborts >= 1
        async with websockets.connect(f"{rig.ws_base}/v1/realtime") as ws_next:
            await _send_start(ws_next)
            assert (await recv_json(ws_next))["type"] == "ready"
            await ws_next.send(json.dumps({"type": "stop"}))
        print("mid-round disconnect → abort + slot freed + next round OK")
    finally:
        await rig.stop()
        fake.close()


def test_vision_disconnect_cleanup() -> None:
    asyncio.run(_test_vision_disconnect_cleanup())


# --------------------------------------------------- lenient start contract


async def _test_vision_lenient_start() -> None:
    fake = FakeSglangOmniServer().start()
    rig = await start_vision(fake)
    try:
        # unknown fields / unusual-but-legal combinations must NOT reject
        async with websockets.connect(f"{rig.ws_base}/v1/realtime") as ws:
            await _send_start(ws, future_field={"x": 1}, do_sample=True,
                              max_tokens_per_second=None)
            assert (await recv_json(ws))["type"] == "ready"
            cfg = fake.configure_payload
            assert cfg["temperature"] == 0.2  # do_sample=true keeps temperature
            await ws.send(json.dumps({"type": "stop"}))
        await asyncio.sleep(0.2)

        # empty prompt IS a rejection
        async with websockets.connect(f"{rig.ws_base}/v1/realtime") as ws:
            await _send_start(ws, prompt="   ")
            err = await recv_json(ws)
            assert err["type"] == "error" and "prompt" in err["message"], err
        # a non-start first message likewise
        async with websockets.connect(f"{rig.ws_base}/v1/realtime") as ws:
            await ws.send(json.dumps({"type": "hello"}))
            err = await recv_json(ws)
            assert err["type"] == "error" and "start" in err["message"], err
        assert rig.pool.busy == 0
        print("lenient start contract (unknown fields OK, empty prompt rejected): OK")
    finally:
        await rig.stop()
        fake.close()


def test_vision_lenient_start() -> None:
    asyncio.run(_test_vision_lenient_start())


# ------------------------------------------------------ standalone app factory


async def _test_vision_standalone_failfast() -> None:
    # without SGLANG_OMNI_URLS the lifespan must refuse to start (fail fast);
    # asserted against the lifespan context directly — uvicorn turns the
    # failure into a process-level SystemExit, which a task cannot carry
    import pytest

    from server.gateway.vision import create_vision_app

    app = create_vision_app(Settings(sglang_omni_urls=""))
    with pytest.raises(RuntimeError, match="SGLANG_OMNI_URLS"):
        async with app.router.lifespan_context(app):
            pass  # pragma: no cover — the context never yields
    print("standalone app fail-fast without SGLANG_OMNI_URLS: OK")


def test_vision_standalone_failfast() -> None:
    asyncio.run(_test_vision_standalone_failfast())


async def _test_vision_standalone_e2e() -> None:
    # the factory serves the full §5 protocol by itself
    from server.gateway.vision import create_vision_app

    fake = FakeSglangOmniServer().start()
    app = create_vision_app(Settings(
        sglang_omni_urls=fake.url, sglang_omni_connect_timeout_s=5.0,
        sglang_omni_health_interval_s=600.0, vision_start_timeout_s=5.0,
        vision_frame_timeout_s=6.0, vision_round_timeout_s=30.0))
    config = uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning")
    server = uvicorn.Server(config)
    task = asyncio.create_task(server.serve())
    try:
        while not server.started:
            await asyncio.sleep(0.02)
            if task.done():
                raise AssertionError(f"uvicorn failed to start: {task.exception()}")
        port = server.servers[0].sockets[0].getsockname()[1]
        async with websockets.connect(f"ws://127.0.0.1:{port}/v1/realtime") as ws:
            await _send_start(ws)
            assert (await recv_json(ws))["type"] == "ready"
            await _upload_frame(ws, timestamp=1.0)
            assert (await recv_json(ws)) == {"type": "frame_ack"}
            fake.send_delta("画面中有")
            fake.send_silence()
            assert (await recv_json(ws))["text"] == "画面中有"
            assert (await recv_json(ws))["text"] == "<|im_end|>"
            await ws.send(json.dumps({"type": "stop"}))
    finally:
        server.should_exit = True
        await asyncio.gather(task, return_exceptions=True)
        fake.close()
    print("standalone app serves the §5 protocol end-to-end: OK")


def test_vision_standalone_e2e() -> None:
    asyncio.run(_test_vision_standalone_e2e())
