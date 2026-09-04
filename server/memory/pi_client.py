"""Thin HTTP client for the pi_agent side of the memory protocol (board parity).

The gateway calls pi_agent for two LLM jobs: `/decide` (should this turn
retrieve memory at all) and `/compact` (compress the rollover journal into
{summary, pins}). stdlib-only (urlopen) — no new dependency. Every failure
mode (timeout, non-200, bad payload) returns None so the caller degrades to
the local-vector / verbatim-tail behavior without disturbing the turn, and is
logged at WARNING level throttled per endpoint (60s) to avoid per-turn spam.
"""
from __future__ import annotations

import json
import socket
import time
from typing import Any, Dict, Optional
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from ..config import Settings
from ..logging_conf import get_logger

log = get_logger(__name__)

_WARN_EVERY_S = 60.0
_last_warn_at: Dict[str, float] = {}
_MAX_ATTEMPTS = 3  # 最多三次，含首次
_BACKOFF_S = (0.3, 0.8)


def _warn_throttled(url: str, message: str, *args: Any) -> None:
    now = time.monotonic()
    if now - _last_warn_at.get(url, 0.0) < _WARN_EVERY_S:
        return
    _last_warn_at[url] = now
    log.warning("pi_agent call %s failed: " + message + " (further failures suppressed for %ds)",
                url, *args, int(_WARN_EVERY_S))


def _post_json(url: str, payload: Dict[str, Any], timeout: float) -> Optional[Dict[str, Any]]:
    """POST JSON with up to 3 attempts (timeout/connection/5xx retry, 4xx hard-fail)."""
    last_error: Optional[str] = None
    for attempt in range(_MAX_ATTEMPTS):
        request = Request(
            url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=timeout) as response:
                status = int(response.status)
                if 200 <= status < 300:
                    body = json.loads(response.read().decode("utf-8"))
                    if isinstance(body, dict):
                        return body
                    last_error = "non-object JSON payload"
                    _warn_throttled(url, "%s (no retry)", last_error)
                    return None  # 格式错误重试无意义
                last_error = f"HTTP {status}"
                if 400 <= status < 500:
                    _warn_throttled(url, "%s (no retry)", last_error)
                    return None  # 4xx 是请求问题，重试无意义
        except (OSError, URLError, ValueError) as exc:
            last_error = str(exc)
        if attempt < _MAX_ATTEMPTS - 1:
            log.info("pi_agent call %s attempt %d/%d failed (%s); retrying",
                     url, attempt + 1, _MAX_ATTEMPTS, last_error)
            time.sleep(_BACKOFF_S[attempt])
    _warn_throttled(url, "%s (exhausted %d attempts)", last_error, _MAX_ATTEMPTS)
    return None


class PiAgentClient:
    """Stateless caller; constructed from Settings, shared per session/manager."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._url = (settings.memory_pi_url or "").rstrip("/")
        self._reachability: tuple = (0.0, False)

    def decide(
        self,
        conversation_id: str,
        recent_turns: str,
        pending_user_text: str,
    ) -> Optional[Dict[str, Any]]:
        """Ask pi_agent whether to retrieve; None means degrade to local gating."""
        if not self._url:
            return None
        return _post_json(
            f"{self._url}/decide",
            {
                "conversation_id": str(conversation_id),
                "recent_turns": recent_turns,
                "pending_user_text": pending_user_text,
            },
            float(self._settings.memory_pi_decide_timeout_s),
        )

    def compact(self, conversation_id: str, journal: str) -> Optional[Dict[str, Any]]:
        """Ask pi_agent to compress the journal; None means caller must fall back."""
        if not self._url:
            return None
        return _post_json(
            f"{self._url}/compact",
            {"conversation_id": str(conversation_id), "journal": journal},
            float(self._settings.memory_pi_compact_timeout_s),
        )

    def reachable(self, cache_seconds: float = 5.0) -> bool:
        """Best-effort TCP reachability probe, cached for a few seconds."""
        if not self._url:
            return False
        cached_at, cached = self._reachability
        if time.monotonic() - cached_at < cache_seconds:
            return cached
        parsed = urlparse(self._url)
        ok = False
        try:
            with socket.create_connection(
                (parsed.hostname or "127.0.0.1", parsed.port or 80), timeout=0.5
            ):
                ok = True
        except OSError:
            ok = False
        self._reachability = (time.monotonic(), ok)
        return ok
