"""sglang-omni replica pool for the gateway plane.

Same READY/BUSY/DOWN discipline as adapters/vlm/moss_vl_sglang_omni/pool.py but
deliberately thinner: this plane owns the WS handshake itself (session.py), so
the pool only does slot bookkeeping, health probing, and capacity accounting.

Capacity model: each replica hosts up to `sglang_omni_sessions_per_replica`
live sessions (the remote omni instance's --max-running-requests; the two MUST
match). READY = healthy with a free slot, BUSY = healthy but full, DOWN =
unhealthy/quarantined.

- acquire() reserves a slot on the READY replica with the FEWEST used slots
  (ties → lowest index; raises GatewayCapacityError when none); the caller
  runs the WS handshake and MUST pair every acquire with exactly one
  release().
- mark_full(index): omni rejected the handshake with
  session_capacity_exceeded — that instance is full server-side (it may host
  sessions we don't track), so acquire skips it; the prober clears the mark
  once /health answers again.
- release(index, transport_dead=True) quarantines the replica (DOWN) because
  a dead transport means the server may be wedged; the prober flips it back
  once GET /health answers again (READY if a slot is free, else BUSY).
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import requests

from ..config import Settings
from ..logging_conf import get_logger

log = get_logger(__name__)

READY, BUSY, DOWN = "READY", "BUSY", "DOWN"


class GatewayCapacityError(RuntimeError):
    """No READY replica slot — the REST layer maps this to 503."""

    def __init__(self, capacity: int, busy: int):
        super().__init__(f"no READY sglang-omni replica ({busy}/{capacity} slots busy)")
        self.capacity = capacity
        self.busy = busy


@dataclass
class GatewayReplica:
    url: str
    state: str = READY
    slots: int = 1            # sessions this replica may host (= omni --max-running-requests)
    used: int = 0             # slots held by live tracked sessions
    capacity_limited: bool = False  # omni reported session_capacity_exceeded;
                                    # the prober re-checks and clears the mark
    health: Dict[str, Any] = field(default_factory=dict)


class GatewayPool:
    def __init__(self, settings: Settings):
        self.s = settings
        urls = [u.strip().rstrip("/")
                for u in str(settings.sglang_omni_urls or "").split(",") if u.strip()]
        slots = max(1, int(settings.sglang_omni_sessions_per_replica or 1))
        self._replicas: List[GatewayReplica] = [
            GatewayReplica(url=u, slots=slots) for u in urls]
        self._lock = threading.Lock()
        self._prober_stop = threading.Event()
        self._prober: Optional[threading.Thread] = None

    # ------------------------------------------------------------ introspection

    @property
    def capacity(self) -> int:
        return sum(r.slots for r in self._replicas)

    @property
    def busy(self) -> int:
        return sum(r.used for r in self._replicas)

    @property
    def replicas(self) -> List[GatewayReplica]:
        return self._replicas

    def replica_url(self, index: int) -> str:
        return self._replicas[index].url

    def first_ready_url(self) -> Optional[str]:
        """First READY replica, else a BUSY one (a live session does not make a
        replica unable to answer GET /v1/models). DOWN replicas never serve."""
        with self._lock:
            for state in (READY, BUSY):
                for r in self._replicas:
                    if r.state == state:
                        return r.url
        return None

    def status(self) -> Dict[str, Any]:
        with self._lock:
            replicas = [{"url": r.url, "state": r.state} for r in self._replicas]
            return {
                "instances": len(self._replicas),
                "active_sessions": self.busy,
                "capacity": self.capacity,
                "replicas": replicas,
            }

    def gauge_counts(self) -> tuple:
        """(slots_total, slots_used, replicas_down) for the P4 metrics gauges.
        The pool is the source of truth; the registry scrapes at read time."""
        with self._lock:
            return (sum(r.slots for r in self._replicas),
                    sum(r.used for r in self._replicas),
                    sum(1 for r in self._replicas if r.state == DOWN))

    # ------------------------------------------------------------ slots

    def acquire(self) -> int:
        """Reserve a slot on the least-loaded READY replica (ties → lowest
        index); returns its index."""
        with self._lock:
            picked: Optional[int] = None
            for i, r in enumerate(self._replicas):
                if r.state == READY and r.used < r.slots and (
                        picked is None or r.used < self._replicas[picked].used):
                    picked = i
            if picked is None:
                raise GatewayCapacityError(self.capacity, self.busy)
            r = self._replicas[picked]
            r.used += 1
            if r.used >= r.slots:
                r.state = BUSY
            return picked

    def mark_full(self, index: int) -> None:
        """omni rejected the handshake with session_capacity_exceeded: the
        instance is full server-side (sessions we don't track hold slots) —
        hand back the failed reservation and mark the replica BUSY so acquire
        skips it. The prober re-probes marked replicas and clears the mark once
        /health answers, so a transient server-side full state cannot wedge
        the replica forever."""
        with self._lock:
            r = self._replicas[index]
            r.used = max(0, r.used - 1)  # the rejected acquire held nothing
            r.capacity_limited = True
            r.state = BUSY
        log.info("gateway replica %d (%s) at session capacity — marked full",
                 index, r.url)

    def release(self, index: int, transport_dead: bool = False) -> None:
        with self._lock:
            r = self._replicas[index]
            r.used = max(0, r.used - 1)
            r.capacity_limited = False  # a tracked slot drained — fullness may have eased
            # a dead transport means the server may be wedged — quarantine the
            # replica until the prober's next health poll clears it
            r.state = DOWN if transport_dead else (READY if r.used < r.slots else BUSY)
        log.info("gateway replica %d (%s) released%s", index, r.url,
                 " (transport dead — quarantined)" if transport_dead else "")

    # ------------------------------------------------------------ health

    def _probe_health(self, url: str) -> Optional[Dict[str, Any]]:
        try:
            resp = requests.get(f"{url}/health",
                                timeout=max(1.0, self.s.sglang_omni_connect_timeout_s))
            if resp.ok:
                return resp.json() if resp.content else {"ok": True}
        except requests.RequestException:
            pass
        return None

    def start_prober(self) -> None:
        if self._prober is not None:
            return
        interval = max(0.2, float(self.s.sglang_omni_health_interval_s))

        def prober() -> None:
            while not self._prober_stop.wait(interval):
                for i, r in enumerate(self._replicas):
                    with self._lock:
                        needs_probe = r.state == DOWN or r.capacity_limited
                    if not needs_probe:
                        continue
                    health = self._probe_health(r.url)
                    if health is None:
                        continue
                    with self._lock:
                        r.health = health
                        if r.state == DOWN:
                            r.state = READY if r.used < r.slots else BUSY
                            log.info("gateway replica %d recovered (%s)", i, r.url)
                        elif r.capacity_limited:
                            # server healthy again — the sessions that filled it
                            # may have drained; a wrong clear self-corrects on
                            # the next capacity rejection
                            r.capacity_limited = False
                            r.state = READY if r.used < r.slots else BUSY
                            log.info("gateway replica %d capacity mark cleared (%s)",
                                     i, r.url)

        self._prober = threading.Thread(
            target=prober, name="gateway-omni-health", daemon=True)
        self._prober.start()

    def close(self) -> None:
        self._prober_stop.set()
        if self._prober is not None:
            self._prober.join(timeout=2.0)
            self._prober = None
