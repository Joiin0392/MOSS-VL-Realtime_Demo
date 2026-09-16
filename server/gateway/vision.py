"""Standalone vision protocol plane: WS /v1/realtime?session_id=...

调用端到模型的实时视觉协议 (§5): the vision caller speaks a one-shot analysis
protocol directly against this service — no REST session, no ws_token, no auth
headers (an empty `session_id=` query parameter is accepted and ignored; model
selection is by deployment address, `start` carries no model field).

    caller                          this file                       sglang-omni
    ---- start (prompt+params) -->  map → session.configure ------>
    <-- {"type":"ready"} ---------  after session.ready
    ---- frame meta + JPEG ------>  two-phase put_frame
    <-- {"type":"frame_ack"} ----  per frame (batch arrivals OK)
    <-- {"type":"output", text} --  deltas, control tokens stripped
    <-- {"type":"output","<|im_end|>"}  on response.turn.silence
    ---- stop ------------------->  session.abort → slot released → close

Error shape: {"type": "error", "message": "..."} — no code field. A busy
service returns a message containing "realtime session is already active"
(the caller's retry trigger). The end marker is only sent after at least one
non-empty visible output chunk (a bare marker would be judged "no valid
output" by the caller). Sessions are isolated per connection: stop, error, or
disconnect always releases the replica slot and allows the next round.

Param mapping (start → session.configure; omni is extra=forbid):
    prompt                  → prompt (required, non-empty)
    max_new_tokens          → max_new_tokens            (clamped: vision_max_new_tokens)
    max_tokens_per_second   → max_tokens_per_turn       (omni tokens/second rate cap)
    do_sample=false         → temperature forced 0.0 (greedy), temperature ignored
    temperature/top_p       → temperature/top_p          (when do_sample)
    frame_queue_size        → input_queue_capacity       (clamped: vision_max_input_queue)
    top_k/repetition_penalty→ ignored (logged once; never a rejection reason)

The FINAL output (the end marker) is emitted after every submitted frame has
been acknowledged whenever generation order allows it; interleaving of output
with frame_ack is protocol-legal either way.
"""
from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Dict, Optional, Tuple

from fastapi import APIRouter, FastAPI, WebSocket

from ..adapters.vlm.moss_vl_hf.online_pool import NoFreeReplica
from ..adapters.vlm.moss_vl_sglang_omni.pool import SglangOmniPool
from ..logging_conf import get_logger

log = get_logger(__name__)
router = APIRouter(tags=["gateway"])

BUSY_MESSAGE = "realtime session is already active"
END_MARKER = "<|im_end|>"
# demo-plane control tokens the vision protocol must never leak; the turn-end
# ones are translated into the end marker instead
ROUND_START_TOKEN = "<|round_start|>"
TURN_END_TOKENS = {"<|silence|>", "<|eot_id|>", "<|round_end|>"}
WS_CLOSE_NORMAL = 1000

# message types the SERVER owns — a client sending them is confused, but the
# protocol asks us to be lenient rather than rejecting rounds outright
SERVER_OWNED_TYPES = {"ready", "frame_ack", "output", "session_end"}


def get_vision_pool(conn: Any) -> SglangOmniPool:
    """The pool lives on app.state (built in the lifespan / injected by tests)."""
    return conn.app.state.vision_pool


# ------------------------------------------------------------------ helpers


async def _send(ws: WebSocket, payload: Dict[str, Any]) -> bool:
    try:
        await ws.send_text(json.dumps(payload, ensure_ascii=False))
        return True
    except Exception:  # noqa: BLE001 — peer vanished mid-send
        return False


async def _send_error(ws: WebSocket, message: str) -> None:
    log.info("vision round error: %s", message)
    await _send(ws, {"type": "error", "message": message})


async def _recv(ws: WebSocket, timeout: float) -> Optional[Dict[str, Any]]:
    """One raw ASGI message, or None on timeout. Disconnects surface as
    {"type": "websocket.disconnect"} through the normal return path."""
    try:
        return await asyncio.wait_for(ws.receive(), timeout)
    except asyncio.TimeoutError:
        return None


def _start_params(raw: Dict[str, Any], pool: SglangOmniPool) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Validate + map the start payload onto SglangOmniPool session params.

    Lenient by contract: unknown fields are ignored, and only a missing/empty
    prompt (or non-numeric numerics) reject the round. Returns
    (params, None) or (None, error_message).
    """
    s = pool.s
    prompt = raw.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        return None, "start.prompt must be a non-empty string"

    def _num(key: str, default: float, minimum: float, maximum: float) -> float:
        value = raw.get(key)
        try:
            value = float(default if value is None else value)
        except (TypeError, ValueError):
            raise ValueError(f"start.{key} must be a number")
        return max(minimum, min(maximum, value))

    try:
        max_new_tokens = int(_num("max_new_tokens", 512, 1, s.vision_max_new_tokens))
        rate = _num("max_tokens_per_second", 86400.0, 0.1, 86400.0)
        top_p = _num("top_p", 1.0, 1e-6, 1.0)
        temperature = _num("temperature", 0.0, 0.0, 2.0)
        queue = int(_num("frame_queue_size", 8, 1, s.vision_max_input_queue))
    except ValueError as exc:
        return None, str(exc)

    do_sample = bool(raw.get("do_sample", True))
    # frame_queue_size maps onto the omni input queue so one whole batch fits
    # in flight (the caller uploads all frames before reading acks)
    return {
        "prompt": prompt.strip(),
        "max_new_tokens": max_new_tokens,
        "max_tokens_per_turn": rate,
        "do_sample": do_sample,
        "temperature": temperature,
        "top_p": top_p,
        # passed through so the pool's unsupported-param log names them once —
        # the mapping layer never rejects on them
        "top_k": raw.get("top_k"),
        "repetition_penalty": raw.get("repetition_penalty"),
        "frame_queue_size": raw.get("frame_queue_size"),
        "input_queue_capacity": queue,
    }, None


class _Round:
    """State shared by the input loop and the output pump."""

    def __init__(self) -> None:
        self.visible = False        # ≥1 non-empty output chunk sent
        self.completed = False      # end marker sent
        self.frames_acked = 0
        self.stop_seen = False


# ------------------------------------------------------------------ pumps


async def _output_pump(ws: WebSocket, session: Any, round_: _Round) -> Optional[str]:
    """Session output queue → output messages. Returns an error message when
    the round must end without the caller's stop (fatal/no active session)."""
    while True:
        batch = await asyncio.to_thread(session.poll_output, 0.2, 64)
        for text, event in zip(batch.chunks, batch.chunk_events):
            if str(event.get("sglang_event_type")) == "error" or text.startswith("[ERROR] "):
                return str(text).removeprefix("[ERROR] ").strip() or "vision session failed"
            if text == ROUND_START_TOKEN:
                continue
            if text in TURN_END_TOKENS:
                if not round_.completed:
                    round_.completed = True
                    if round_.visible and not await _send(
                            ws, {"type": "output", "text": END_MARKER}):
                        return None  # peer gone; the input loop will notice
                continue
            if text:
                round_.visible = True
                if not await _send(ws, {"type": "output", "text": str(text)}):
                    return None
        if not batch.active:
            # session over (done/abort/transport death). A completed round just
            # waits for the caller's stop; anything else ends the round here.
            if round_.completed or not round_.visible:
                if not round_.completed and round_.visible:
                    round_.completed = True
                    await _send(ws, {"type": "output", "text": END_MARKER})
                if not round_.visible and not round_.completed:
                    return "vision session ended without any output"
                return None
            # visible text but no turn-end marker ever arrived: close the round
            # ourselves so the caller is not stuck waiting for the marker
            round_.completed = True
            await _send(ws, {"type": "output", "text": END_MARKER})
            return None


async def _input_loop(ws: WebSocket, session: Any, pool: SglangOmniPool,
                      round_: _Round) -> Optional[str]:
    """Caller → session: frame pairs and stop. Returns an error message on a
    protocol failure, None on stop/disconnect."""
    s = pool.s
    while True:
        message = await _recv(ws, s.vision_frame_timeout_s)
        if message is None:
            return "timed out waiting for the next client message"
        if message.get("type") == "websocket.disconnect":
            return None
        text = message.get("text")
        if text is None:
            if message.get("bytes") is not None:
                return "unexpected binary message (send a frame metadata message first)"
            continue
        try:
            payload = json.loads(text)
            assert isinstance(payload, dict)
        except (ValueError, AssertionError):
            return "malformed JSON message"
        mtype = str(payload.get("type") or "")
        if mtype == "stop":
            round_.stop_seen = True
            return None
        if mtype == "frame":
            binary = await _recv(ws, s.vision_frame_timeout_s)
            if binary is None:
                return "timed out waiting for the frame binary payload"
            if binary.get("type") == "websocket.disconnect":
                return None
            data = binary.get("bytes")
            if data is None:
                return "frame metadata must be immediately followed by a binary JPEG message"
            try:
                timestamp = float(payload.get("timestamp"))
            except (TypeError, ValueError):
                return "frame.timestamp must be a number (seconds in the media)"
            try:
                status = await asyncio.to_thread(
                    session.put_frame, data, timestamp, len(data))
            except ValueError as exc:
                return f"frame rejected: {exc}"
            except Exception as exc:  # noqa: BLE001 — session died mid-frame
                return f"frame submission failed: {exc}"
            if status.get("frame_dropped"):
                return ("input queue overflow: frame dropped by the backend "
                        f"({status.get('drop_reason')})")
            round_.frames_acked += 1
            if not await _send(ws, {"type": "frame_ack"}):
                return None
            continue
        if mtype in SERVER_OWNED_TYPES:
            return f"unexpected message type from caller: {mtype}"
        # unknown types are ignored (lenient contract), start after the first
        # message is likewise tolerated
        log.debug("vision round: ignoring message type %r", mtype)


# ------------------------------------------------------------------ endpoint


@router.websocket("/v1/realtime")
async def vision_realtime_ws(websocket: WebSocket):
    await websocket.accept()
    pool = get_vision_pool(websocket)
    s = pool.s

    # ---- start (the only mandatory pre-session exchange) ----
    message = await _recv(websocket, s.vision_start_timeout_s)
    if message is None or message.get("type") == "websocket.disconnect":
        return
    raw: Optional[Dict[str, Any]] = None
    if message.get("text"):
        try:
            parsed = json.loads(message["text"])
            raw = parsed if isinstance(parsed, dict) else None
        except ValueError:
            raw = None
    if raw is None or str(raw.get("type") or "") != "start":
        await _send_error(websocket, "expected a start message first")
        await websocket.close(code=WS_CLOSE_NORMAL)
        return
    params, error = _start_params(raw, pool)
    if params is None:
        await _send_error(websocket, error or "invalid start message")
        await websocket.close(code=WS_CLOSE_NORMAL)
        return

    # ---- session (configure included); busy → the caller's retry trigger ----
    started = time.monotonic()
    try:
        session = await asyncio.to_thread(pool.start_realtime_session, **params)
    except NoFreeReplica:
        log.info("vision round rejected: no free sglang-omni replica slot")
        await _send_error(websocket, BUSY_MESSAGE)
        await websocket.close(code=WS_CLOSE_NORMAL)
        return
    except Exception as exc:  # noqa: BLE001 — handshake/configure failures
        log.warning("vision round session start failed: %s", exc)
        await _send_error(websocket, f"vision session start failed: {exc}")
        await websocket.close(code=WS_CLOSE_NORMAL)
        return
    log.info("vision round session %s ready in %.2fs (session_id=%s)",
             session.session_id, time.monotonic() - started, params["prompt"][:32])
    if not await _send(websocket, {"type": "ready"}):
        await asyncio.to_thread(session.stop, 5.0)
        return

    # ---- duplex round ----
    round_ = _Round()
    inputs = asyncio.create_task(_input_loop(websocket, session, pool, round_))
    outputs = asyncio.create_task(_output_pump(websocket, session, round_))
    failure: Optional[str] = None
    try:
        async with asyncio.timeout(s.vision_round_timeout_s):
            await asyncio.wait({inputs, outputs}, return_when=asyncio.FIRST_COMPLETED)
    except TimeoutError:
        failure = "vision round exceeded the server time budget"
    except asyncio.CancelledError:  # app shutdown mid-round
        for task in (inputs, outputs):
            task.cancel()
        await asyncio.gather(inputs, outputs, return_exceptions=True)
        await asyncio.to_thread(session.stop, 2.0)
        raise
    finally:
        # cancel() on an already-finished task is a no-op; the pending twin
        # (still waiting for stop/disconnect) must not outlive the round
        for task in (inputs, outputs):
            task.cancel()
        await asyncio.gather(inputs, outputs, return_exceptions=True)
    for task in (inputs, outputs):
        if task.cancelled():
            continue
        exc = task.exception()
        if exc is not None and not isinstance(exc, asyncio.CancelledError):
            failure = failure or f"vision round pump failed: {exc}"
        elif task is inputs and task.result() and not failure:
            failure = task.result()
        elif task is outputs and task.result() and not failure:
            failure = task.result()
    await asyncio.to_thread(session.stop, 5.0)
    if failure is not None:
        await _send_error(websocket, failure)
    try:
        await websocket.close(code=WS_CLOSE_NORMAL)
    except Exception:  # noqa: BLE001 — peer may already be gone
        pass
    log.info("vision round done: frames=%d visible=%s completed=%s stop=%s failure=%s",
             round_.frames_acked, round_.visible, round_.completed,
             round_.stop_seen, failure)


@router.get("/v1/realtime/health")
async def vision_health(request: Any):
    """Ops surface: replica states + slot water level (gateway REST is not
    mounted in vision standalone mode, so this plane ships its own)."""
    pool = get_vision_pool(request)
    return pool.status()


# ------------------------------------------------------------------ standalone app


def create_vision_app(settings: Optional[Any] = None) -> FastAPI:
    """The standalone VL realtime service: ONLY the vision protocol plane —
    no demo orchestrator, no ASR/TTS sidecars, no GPU placement, no frontend.

    Deploy (any branch whose Settings carries the sglang-omni fields — i.e.
    mainline after the npu/full merge):

        scripts/deploy/run_vision.sh        # or directly:
        <venv>/python -m uvicorn server.gateway.vision:app \
            --host 0.0.0.0 --port 8010 --ws-max-size 67108864

    The replica pool is health-gated in the lifespan (fail-fast: no reachable
    sglang-omni replica → the service refuses to start).
    """
    from contextlib import asynccontextmanager

    from ..config import get_settings
    from ..logging_conf import configure_logging

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        resolved = settings if settings is not None else get_settings()
        configure_logging()
        if not resolved.sglang_omni_urls.strip():
            raise RuntimeError(
                "vision standalone service requires SGLANG_OMNI_URLS "
                "(comma-separated sglang-omni realtime server base URLs)")
        pool = SglangOmniPool(resolved)
        await asyncio.to_thread(pool.load, "", -1, "online_streaming")
        app.state.vision_pool = pool
        log.info("Vision standalone plane up: %d replica(s), %d slot(s)",
                 len(pool.replicas), pool.capacity)
        try:
            yield
        finally:
            for replica in pool.replicas:
                for inner in list(replica.sessions):
                    try:
                        await asyncio.to_thread(inner.stop, 2.0)
                    except Exception as exc:  # noqa: BLE001
                        log.warning("vision session stop failed: %s", exc)

    app = FastAPI(title="MOSS-VL Vision Realtime", version="0.1.0",
                  lifespan=lifespan)
    app.include_router(router)
    return app


app = create_vision_app()
