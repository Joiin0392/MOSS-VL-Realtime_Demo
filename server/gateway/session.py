"""GatewaySession: one external realtime session on the gateway plane.

Lifecycle (GATEWAY_PLAN.md §2-P1):

  REST create → pool.acquire() → SglangOmniClient.open() (the returned
  `session.created` is cached and replayed first on attach) → receiver thread
  streams omni events into an asyncio.Queue (via loop.call_soon_threadsafe)
  → WSS attach drains the queue verbatim to the client → either side ending
  destroys the session (this plane has NO grace/reconnect: omni cannot resume
  a session, so a client disconnect is final).

Passthrough discipline: client→omni text frames and omni→client events cross
as raw text, never re-serialized (SglangOmniClient.start_receiver(pass_raw=True)
/ send_text). State tracking for GET /v1/realtime/sessions/{id} parses its own
copy of each event and never touches the forwarded bytes.

Frame-size policing lives here because omni only *advertises* max_frame_bytes
(in session.configured) without enforcing it: an inbound binary frame over
settings.gateway_max_frame_bytes gets error{code:invalid_request} + close 1009.
"""
from __future__ import annotations

import asyncio
import json
import threading
import time
import uuid
from typing import Any, Dict, Optional

from fastapi import WebSocket

from ..adapters.vlm.moss_vl_sglang_omni.client import (
    SessionCapacityExceeded, SglangOmniClient)
from ..config import Settings
from ..logging_conf import get_logger
from .metrics import (
    COUNTER_ABNORMAL_DISCONNECTS, COUNTER_ATTACH_TIMEOUTS,
    COUNTER_FRAMES_ACCEPTED, COUNTER_SESSIONS_CREATED, COUNTER_TEXT_CHARS,
    END_ATTACH_TIMEOUT, END_CLIENT_DISCONNECT, END_OMNI_DEAD, END_RESET,
    END_SESSION_DONE, END_SHUTDOWN, GAUGE_ACTIVE_SESSIONS, GatewayMetrics, UsageLog)
from .pool import GatewayCapacityError, GatewayPool
from .tokens import TokenIssuer

log = get_logger(__name__)

# phases reported by GET /v1/realtime/sessions/{id}
PHASES = ("created", "configured", "ready", "streaming", "parked", "done")

# close codes this plane uses on the client socket
WS_CLOSE_POLICY = 1008        # token/attach policy violations (ws.py)
WS_CLOSE_TOO_BIG = 1009       # frame exceeds gateway_max_frame_bytes
WS_CLOSE_NORMAL = 1000        # DELETE / janitor teardown
WS_CLOSE_OMNI_DEAD = 1011     # downstream transport died mid-session
WS_CLOSE_RESET = 1012         # session reset while a client was attached


class GatewaySession:
    """Owns one omni connection, one replica slot, and at most one client WS."""

    def __init__(self, registry: "GatewayRegistry", session_id: str):
        self.registry = registry
        self.session_id = session_id
        # platform-facing trace id (P3): one per session, kept across resets;
        # paired with session_id + the omni request_id in logs and the ledger
        self.trace_id = uuid.uuid4().hex
        self.created_at = time.time()
        self.request_id: Optional[str] = None
        self.model: Optional[str] = None

        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._index = -1                    # pool replica index
        self._client: Optional[SglangOmniClient] = None
        self._queue: Optional[asyncio.Queue] = None
        self._epoch = 0                     # bumped by reset(); stale receivers/pumps no-op
        self._ws_close_code = WS_CLOSE_NORMAL
        self._lock = threading.Lock()       # guards tracker + attached/destroyed flags
        self._attached = False
        self._destroyed = False
        # tracker (written by the receiver thread, read by REST handlers)
        self._phase = "created"
        self._turn_id = 0
        self._frames_accepted = 0
        self._prompts = 0
        self._text_deltas = 0
        self._text_chars = 0

    # ------------------------------------------------------------ properties

    @property
    def attached(self) -> bool:
        with self._lock:
            return self._attached

    @property
    def destroyed(self) -> bool:
        with self._lock:
            return self._destroyed

    # ------------------------------------------------------------ creation

    async def open_on_replica(self, index: int) -> None:
        """Acquire-side half of create(): WS handshake + receiver on replica `index`."""
        self._loop = asyncio.get_running_loop()
        self._queue = asyncio.Queue()
        await self._open_client(index, epoch=0)

    async def _open_client(self, index: int, epoch: int) -> None:
        """Open the omni connection and start its receiver. The consumed
        `session.created` is re-serialized (semantically identical, key order
        preserved) and queued BEFORE the receiver starts, so attach always
        replays it first; every event after it crosses verbatim (pass_raw)."""
        s = self.registry.settings
        client = SglangOmniClient(self.registry.pool.replica_url(index),
                                  s.sglang_omni_connect_timeout_s)
        created = await asyncio.to_thread(client.open)  # consumes session.created
        self._index = index
        self._client = client
        self.request_id = created.get("request_id")
        self.model = created.get("model")
        self._track(created)
        self._queue.put_nowait(json.dumps(created, ensure_ascii=False))
        client.start_receiver(
            lambda raw, parsed: self._on_omni_event(epoch, raw, parsed),
            lambda reason: self._on_omni_close(epoch, reason),
            pass_raw=True)

    # ------------------------------------------------------------ omni → client

    def _on_omni_event(self, epoch: int, raw: str, parsed: Dict[str, Any]) -> None:
        """Receiver-thread callback: track a parsed copy, queue the raw text."""
        if epoch != self._epoch:
            return  # stale receiver from before a reset
        self._track(parsed)
        loop = self._loop
        if loop is not None:
            loop.call_soon_threadsafe(self._queue.put_nowait, raw)

    def _on_omni_close(self, epoch: int, reason: str) -> None:
        """Receiver-thread callback: the omni transport closed.

        A close after session.done (phase "done") is the NORMAL end of the
        wire protocol — the client gets 1000 and the slot frees WITHOUT
        quarantine. Anything else is a mid-session transport death (1011 +
        DOWN). `_track` runs on this same receiver thread before the close
        callback, so reading _phase here is race-free."""
        if epoch != self._epoch or self.destroyed:
            return
        loop = self._loop
        if loop is None:
            return
        if self._phase == "done":
            log.info("gateway session %s: omni closed cleanly after done",
                     self.session_id)
            loop.call_soon_threadsafe(self._schedule_session_done)
            return
        log.warning("gateway session %s: omni transport closed (%s)", self.session_id, reason)
        loop.call_soon_threadsafe(self._schedule_transport_dead)

    def _schedule_session_done(self) -> None:
        self._ws_close_code = WS_CLOSE_NORMAL
        asyncio.create_task(self.adestroy("omni_session_done", transport_dead=False,
                                          end_reason=END_SESSION_DONE))

    def _schedule_transport_dead(self) -> None:
        self._ws_close_code = WS_CLOSE_OMNI_DEAD
        asyncio.create_task(self.adestroy("omni_transport_dead", transport_dead=True,
                                          end_reason=END_OMNI_DEAD))

    async def _pump_out(self, ws: WebSocket) -> None:
        """Queue → client, verbatim. The None sentinel ends the session socket."""
        while True:
            item = await self._queue.get()
            if item is None:
                try:
                    await ws.close(code=self._ws_close_code)
                except Exception:  # noqa: BLE001 — peer may already be gone
                    pass
                return
            await ws.send_text(item)

    # ------------------------------------------------------------ client → omni

    async def _pump_in(self, ws: WebSocket) -> None:
        """Client → omni, verbatim text; binary frames are size-policed first."""
        max_frame = self.registry.settings.gateway_max_frame_bytes
        while True:
            message = await ws.receive()
            if message.get("type") == "websocket.disconnect":
                return
            text = message.get("text")
            if text is not None:
                await asyncio.to_thread(self._client.send_text, text)
                continue
            data = message.get("bytes")
            if data is None:
                continue
            if len(data) > max_frame:
                log.warning("gateway session %s: frame %dB > max %dB — closing 1009",
                            self.session_id, len(data), max_frame)
                await ws.send_text(json.dumps({
                    "type": "error", "code": "invalid_request",
                    "message": "frame exceeds max_frame_bytes"}, ensure_ascii=False))
                try:
                    await ws.close(code=WS_CLOSE_TOO_BIG)
                except Exception:  # noqa: BLE001
                    pass
                return
            await asyncio.to_thread(self._client.send_bytes, data)

    # ------------------------------------------------------------ attach / run

    def try_attach(self) -> bool:
        with self._lock:
            if self._attached or self._destroyed:
                return False
            self._attached = True
            return True

    async def run(self, ws: WebSocket) -> None:
        """Duplex pump until either side ends; the session is then destroyed."""
        epoch = self._epoch
        pump_out = asyncio.create_task(self._pump_out(ws))
        pump_in = asyncio.create_task(self._pump_in(ws))
        try:
            done, pending = await asyncio.wait(
                {pump_out, pump_in}, return_when=asyncio.FIRST_COMPLETED)
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            for task in done:
                exc = task.exception()
                if exc is not None:
                    log.info("gateway session %s pump ended: %s", self.session_id, exc)
        finally:
            with self._lock:
                self._attached = False
            await self.adestroy("client_ws_ended", epoch=epoch,
                                end_reason=END_CLIENT_DISCONNECT)

    # ------------------------------------------------------------ reset

    async def reset(self) -> Dict[str, Any]:
        """Close the omni connection, reopen on the SAME replica, mint a fresh
        ws_token, zero the counters. Same response shape as session create.
        The old connection's meter closes with a `reset` ledger record; the new
        connection starts a fresh meter under the SAME trace_id (P3)."""
        if self.destroyed:
            raise KeyError(self.session_id)
        log.info("gateway session %s reset (trace_id=%s)", self.session_id, self.trace_id)
        self._write_usage(END_RESET)  # close the old meter BEFORE zeroing
        self._epoch += 1  # detach stale receivers/pumps from the old connection
        if self.attached:
            self._ws_close_code = WS_CLOSE_RESET
            self._queue.put_nowait(None)  # end the old run()'s pump-out
            with self._lock:
                self._attached = False
        old = self._client
        if old is not None:
            await asyncio.to_thread(old.close)
        with self._lock:
            self._phase = "created"
            self._turn_id = 0
            self._frames_accepted = 0
            self._prompts = 0
            self._text_deltas = 0
            self._text_chars = 0
        self.created_at = time.time()  # fresh meter + fresh attach window
        self._ws_close_code = WS_CLOSE_NORMAL
        self._queue = asyncio.Queue()  # fresh queue: no stale events, no sentinel
        try:
            await self._open_client(self._index, epoch=self._epoch)
        except Exception as exc:  # noqa: BLE001
            # the replica is likely wedged — quarantine it and drop the session
            self.registry.pool.release(self._index, transport_dead=True)
            with self._lock:
                self._destroyed = True
            self._write_usage(END_OMNI_DEAD)
            self.registry.metrics.inc(COUNTER_ABNORMAL_DISCONNECTS)
            self.registry.metrics.add_gauge(GAUGE_ACTIVE_SESSIONS, -1)
            self.registry.remove(self.session_id)
            raise GatewayCapacityError(self.registry.pool.capacity,
                                       self.registry.pool.busy) from exc
        token = self.registry.tokens.mint(self.session_id)
        return self.create_payload(token)

    # ------------------------------------------------------------ teardown

    def _write_usage(self, end_reason: str) -> None:
        """Append one reconciliation record to the usage ledger (P3). Runs
        only on session-terminal paths — never in the forwarding loop."""
        ended_at = time.time()
        with self._lock:
            record = {
                "trace_id": self.trace_id,
                "session_id": self.session_id,
                "request_id": self.request_id,
                "model": self.model,
                "replica_url": (self.registry.pool.replica_url(self._index)
                                if self._index >= 0 else None),
                "created_at": self.created_at,
                "ended_at": ended_at,
                "duration_s": round(ended_at - self.created_at, 3),
                "frames_accepted": self._frames_accepted,
                "prompts": self._prompts,
                "text_deltas": self._text_deltas,
                # chars, not tokens: the gateway has no tokenizer (plan allows
                # the char-count approximation)
                "text_chars": self._text_chars,
                "end_reason": end_reason,
            }
        self.registry.usage.record(record)

    async def adestroy(self, reason: str, transport_dead: bool = False,
                       epoch: Optional[int] = None,
                       end_reason: str = END_SHUTDOWN) -> None:
        """Idempotent. epoch is set by run(): a stale pump from before a reset
        must not release the slot the reset just re-acquired."""
        with self._lock:
            if epoch is not None and epoch != self._epoch:
                return
            if self._destroyed:
                return
            self._destroyed = True
        log.info("gateway session %s destroyed (%s, trace_id=%s)",
                 self.session_id, reason, self.trace_id)
        metrics = self.registry.metrics
        metrics.add_gauge(GAUGE_ACTIVE_SESSIONS, -1)
        if end_reason == END_OMNI_DEAD:
            metrics.inc(COUNTER_ABNORMAL_DISCONNECTS)
        elif end_reason == END_ATTACH_TIMEOUT:
            metrics.inc(COUNTER_ATTACH_TIMEOUTS)
        self._write_usage(end_reason)
        queue = self._queue
        if queue is not None:
            try:
                queue.put_nowait(None)  # wake pump-out so it closes the client ws
            except Exception:  # noqa: BLE001
                pass
        client = self._client
        if client is not None:
            await asyncio.to_thread(client.close)
        if self._index >= 0:
            self.registry.pool.release(self._index, transport_dead=transport_dead)
        self.registry.remove(self.session_id)

    # ------------------------------------------------------------ introspection

    def create_payload(self, token: str) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "ws_token": token,
            "ws_url": "/v1/realtime",
            "expires_in": int(self.registry.tokens.ttl_s),
            # doc §7: the REST response carries model/version info alongside
            # session.created (snapshot() has it too)
            "model": self.model,
        }

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "session_id": self.session_id,
                "trace_id": self.trace_id,
                "status": self._phase,
                "frames_accepted": self._frames_accepted,
                "prompts": self._prompts,
                "text_deltas": self._text_deltas,
                "text_chars": self._text_chars,
                "turn_id": self._turn_id,
                "created_at": self.created_at,
                "model": self.model,
                "request_id": self.request_id,
                "attached": self._attached,
                "replica": self.registry.pool.replica_url(self._index)
                           if self._index >= 0 else None,
            }

    # ------------------------------------------------------------ state tracker

    def _track(self, ev: Dict[str, Any]) -> None:
        """Phase/counter transitions from a parsed COPY of the event stream.
        Runs on the receiver thread: the metric updates are O(1) counter
        increments under a short lock — no IO on the passthrough path."""
        etype = ev.get("type")
        metrics = self.registry.metrics
        with self._lock:
            turn_id = ev.get("turn_id")
            if isinstance(turn_id, int):
                self._turn_id = max(self._turn_id, turn_id)
            if self._phase == "done":
                return
            if etype == "session.configured":
                self._phase = "configured"
            elif etype == "session.ready":
                self._phase = "ready"
            elif etype == "input.frame.accepted":
                self._frames_accepted += 1
                metrics.inc(COUNTER_FRAMES_ACCEPTED)
                self._phase = "streaming"
            elif etype == "input.prompt.accepted":
                self._prompts += 1
            elif etype == "response.text.delta":
                self._text_deltas += 1
                chars = len(str(ev.get("delta") or ""))
                self._text_chars += chars
                metrics.inc(COUNTER_TEXT_CHARS, chars)
                self._phase = "streaming"
            elif etype == "response.turn.silence":
                self._phase = "parked"
            elif etype in ("response.done", "session.done"):
                self._phase = "done"
            elif etype == "error":
                # omni error events pass through verbatim; count by code (P4)
                metrics.inc_error(str(ev.get("code") or "unknown"))


class GatewayRegistry:
    """session_id → GatewaySession, plus the attach-timeout janitor.

    create() retries across READY replicas: a replica whose WS handshake is
    capacity-rejected is marked full (pool.mark_full) and the next READY one
    is tried; any other handshake failure quarantines the replica (release
    with transport_dead=True). GatewayCapacityError only when nothing READY
    remains.
    """

    def __init__(self, settings: Settings, pool: GatewayPool, tokens: TokenIssuer,
                 metrics: Optional[GatewayMetrics] = None,
                 usage: Optional[UsageLog] = None):
        self.settings = settings
        self.pool = pool
        self.tokens = tokens
        self.metrics = metrics if metrics is not None else GatewayMetrics()
        self.usage = usage if usage is not None else UsageLog.from_settings(settings)
        self._sessions: Dict[str, GatewaySession] = {}
        self._lock = threading.Lock()
        self._janitor: Optional[asyncio.Task] = None

    # ---- sessions ----

    async def create(self) -> GatewaySession:
        self.ensure_janitor()
        while True:
            index = self.pool.acquire()  # raises GatewayCapacityError when full
            session = GatewaySession(self, session_id=f"gws-{uuid.uuid4().hex[:16]}")
            try:
                await session.open_on_replica(index)
            except SessionCapacityExceeded:
                # the instance is full server-side, not wedged — skip it
                log.warning("gateway create: replica %d at session capacity", index)
                self.pool.mark_full(index)
                continue
            except Exception as exc:  # noqa: BLE001 — handshake/open failure
                log.warning("gateway create: replica %d open failed: %s", index, exc)
                self.pool.release(index, transport_dead=True)
                continue
            with self._lock:
                self._sessions[session.session_id] = session
            self.metrics.inc(COUNTER_SESSIONS_CREATED)
            self.metrics.add_gauge(GAUGE_ACTIVE_SESSIONS, 1)
            log.info("gateway session %s → replica %d (%s) trace_id=%s",
                     session.session_id, index, self.pool.replica_url(index),
                     session.trace_id)
            return session

    def get(self, session_id: str) -> Optional[GatewaySession]:
        with self._lock:
            session = self._sessions.get(session_id)
        if session is not None and session.destroyed:
            return None
        return session

    def remove(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)

    def __len__(self) -> int:
        with self._lock:
            return len(self._sessions)

    # ---- janitor ----

    def ensure_janitor(self) -> None:
        """Start the attach-timeout sweeper on the running loop (lazy: tests
        and the app lifespan both end up here on their serving loop)."""
        if self._janitor is None or self._janitor.done():
            self._janitor = asyncio.create_task(self._janitor_loop())

    async def _janitor_loop(self) -> None:
        timeout = max(0.2, float(self.settings.gateway_attach_timeout_s))
        interval = max(0.2, min(5.0, timeout / 2.0))
        while True:
            await asyncio.sleep(interval)
            self.tokens.purge_expired()
            now = time.time()
            with self._lock:
                victims = [s for s in self._sessions.values()
                           if not s.attached and not s.destroyed
                           and now - s.created_at > timeout]
            for session in victims:
                log.info("gateway session %s never attached in %.0fs — destroying "
                         "(trace_id=%s)", session.session_id, timeout, session.trace_id)
                await session.adestroy("attach_timeout", end_reason=END_ATTACH_TIMEOUT)

    # ---- shutdown ----

    async def aclose(self) -> None:
        if self._janitor is not None:
            self._janitor.cancel()
            self._janitor = None
        with self._lock:
            sessions = list(self._sessions.values())
        for session in sessions:
            await session.adestroy("shutdown", end_reason=END_SHUTDOWN)
