"""Sync WebSocket client for one sglang-omni `/v1/video/realtime` connection.

Thin wrapper over `websocket-client` (already in requirements.txt) in the
spirit of board's `SGLangRealtimeSession` transport: blocking handshake, a
single daemon receiver thread, and a lock-serialized send side — nothing here
touches the gateway's asyncio loop.

Wire facts (verified against the sglang-omni source):
  connect → server sends `session.created` (or `error[session_capacity_exceeded]`
  + close 1013) → client sends `session.configure` (exactly once; extra fields
  are forbidden) → server replies `session.configured` then `session.ready`.
  No input is accepted before `session.ready`.
"""
from __future__ import annotations

import json
import threading
import time
from typing import Any, Callable, Dict, Optional, Tuple

from ....logging_conf import get_logger

log = get_logger(__name__)

# inbound deltas are tiny; the cap only guards against a runaway peer
MAX_INBOUND_FRAME_BYTES = 32 * 1024 * 1024


class SessionCapacityExceeded(RuntimeError):
    """The server rejected the session at accept time (close code 1013)."""


def http_to_ws_url(base_url: str) -> str:
    """Derive the realtime WS endpoint from a replica's http(s) base URL."""
    url = base_url.strip().rstrip("/")
    if url.startswith("https://"):
        url = "wss://" + url[len("https://"):]
    elif url.startswith("http://"):
        url = "ws://" + url[len("http://"):]
    elif not url.startswith(("ws://", "wss://")):
        url = "ws://" + url
    return url + "/v1/video/realtime"


class SglangOmniClient:
    """Owns the transport for ONE sglang-omni realtime session."""

    def __init__(self, base_url: str, connect_timeout_s: float = 10.0):
        self.base_url = base_url
        self.ws_url = http_to_ws_url(base_url)
        self.connect_timeout_s = max(1.0, float(connect_timeout_s))
        self._ws: Any = None
        self._send_lock = threading.Lock()
        self._recv_thread: Optional[threading.Thread] = None
        self._closed = threading.Event()

    # ------------------------------------------------------------ handshake

    def open(self) -> Dict[str, Any]:
        """Connect and consume the mandatory first event (`session.created`)."""
        import websocket  # websocket-client (sync API)

        self._ws = websocket.create_connection(
            self.ws_url, timeout=self.connect_timeout_s, enable_multithread=True)
        first = self._recv_event()
        if first.get("type") == "error":
            if first.get("code") == "session_capacity_exceeded":
                self.close()
                raise SessionCapacityExceeded(
                    str(first.get("message") or "session capacity exceeded"))
            self.close()
            raise RuntimeError(str(first.get("message") or "sglang-omni rejected the session"))
        if first.get("type") != "session.created":
            self.close()
            raise RuntimeError(f"expected session.created, got: {first}")
        return first

    def configure(self, payload: Dict[str, Any], timeout_s: float,
                  on_event: Callable[[Dict[str, Any]], None]) -> None:
        """Send `session.configure` and block until `session.ready`.

        Non-handshake events arriving in the configure window (the initial
        prefill can already emit text) are forwarded to `on_event`.
        """
        self.send_json(payload)
        deadline = time.monotonic() + max(1.0, timeout_s)
        configured = False
        ready = False
        while time.monotonic() < deadline and not (configured and ready):
            self._ws.settimeout(max(0.1, deadline - time.monotonic()))
            try:
                message = self._recv_event()
            except Exception as exc:  # noqa: BLE001 — socket.timeout included
                if "timed out" in str(exc).lower():
                    continue
                raise
            event_type = message.get("type")
            if event_type == "session.configured":
                configured = True
            elif event_type == "session.ready":
                ready = True
            elif event_type == "error":
                raise RuntimeError(
                    str(message.get("message") or "sglang-omni session.configure failed"))
            else:
                on_event(message)
        if not configured or not ready:
            raise TimeoutError("sglang-omni session did not become ready in time")

    # ------------------------------------------------------------ io

    def start_receiver(self, on_event: Callable, on_close: Callable[[str], None],
                       pass_raw: bool = False) -> None:
        """Start the daemon receive loop; events stream until the socket dies.

        pass_raw=True (the gateway plane's verbatim passthrough) calls
        on_event(raw_text, parsed); the default calls on_event(parsed).
        """
        self._ws.settimeout(1.0)  # poll for local close while blocking on recv

        def loop() -> None:
            reason = "ws_closed"
            try:
                while not self._closed.is_set():
                    try:
                        raw, message = self._recv_message()
                    except Exception as exc:  # noqa: BLE001
                        if self._closed.is_set():
                            break
                        if "timed out" in str(exc).lower():
                            continue
                        reason = f"ws_closed: {exc}"
                        log.warning("sglang-omni recv failed (%s): %s", self.ws_url, exc)
                        break
                    on_event(raw, message) if pass_raw else on_event(message)
            finally:
                self._closed.set()
                on_close(reason)

        self._recv_thread = threading.Thread(
            target=loop, name=f"sglang-omni-recv-{id(self) & 0xFFFF:04x}", daemon=True)
        self._recv_thread.start()

    def send_json(self, payload: Dict[str, Any]) -> None:
        with self._send_lock:
            self._ws.send(json.dumps(payload, ensure_ascii=False))

    def send_text(self, raw: str) -> None:
        """Forward a client text frame verbatim (gateway plane passthrough)."""
        with self._send_lock:
            self._ws.send(raw)

    def send_bytes(self, data: bytes) -> None:
        with self._send_lock:
            self._ws.send_binary(data)

    def close(self) -> None:
        self._closed.set()
        try:
            if self._ws is not None:
                self._ws.close()
        except Exception:  # noqa: BLE001
            pass
        if self._recv_thread is not None and self._recv_thread is not threading.current_thread():
            self._recv_thread.join(timeout=2.0)

    def _recv_message(self) -> Tuple[str, Dict[str, Any]]:
        """(raw_text, parsed) for one inbound event; raises on binary/close."""
        raw = self._ws.recv()
        if isinstance(raw, (bytes, bytearray)):
            if len(raw) > MAX_INBOUND_FRAME_BYTES:
                raise RuntimeError("oversized inbound frame from sglang-omni")
            raise RuntimeError("unexpected binary event from sglang-omni")
        if not raw:
            raise ConnectionError("sglang-omni WebSocket closed")
        message = json.loads(raw)
        if not isinstance(message, dict):
            raise RuntimeError("non-object event from sglang-omni")
        return raw, message

    def _recv_event(self) -> Dict[str, Any]:
        return self._recv_message()[1]
