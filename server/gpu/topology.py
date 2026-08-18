"""GPU topology probe + attention-impl selection.

Probes via `nvidia-smi` in a subprocess with a watchdog — NOT via torch: a torch
CUDA probe in the gateway would create a CUDA context on every GPU (hundreds of
MiB each) and can hang indefinitely on a wedged driver. A short-lived
`python -c` torch child is the fallback when nvidia-smi is missing.
"""
from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from typing import List, Optional, Tuple

from ..logging_conf import get_logger

log = get_logger(__name__)

_SMI_QUERY = "index,uuid,name,memory.total,memory.used,compute_cap"
_SMI_CMD = [
    "nvidia-smi",
    f"--query-gpu={_SMI_QUERY}",
    "--format=csv,noheader,nounits",
]

# compute capability at/above which flash-attn has no cubins (Blackwell sm_120;
# the deployed flash-attn 2.8.1 wheel was built on H200 → sm_90 only, but sdpa
# is only *forced* where FA categorically cannot run — sm_89/90 keep FA and the
# worker's warmup forward flips to sdpa if the kernel image is missing)
FLASH_ATTN_UNSUPPORTED_CC = (12, 0)


@dataclass(frozen=True)
class GpuInfo:
    index: int
    uuid: str
    name: str
    mem_total_mib: int
    mem_used_mib: int
    compute_cap: Tuple[int, int]

    @property
    def mem_free_mib(self) -> int:
        return max(0, self.mem_total_mib - self.mem_used_mib)


def parse_smi_csv(text: str) -> List[GpuInfo]:
    """Parse `nvidia-smi --query-gpu=... --format=csv,noheader,nounits` output.

    Malformed lines are skipped with a warning (a half-wedged driver can emit
    garbage rows) — the probe returns whatever parsed cleanly.
    """
    gpus: List[GpuInfo] = []
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 6:
            log.warning("nvidia-smi row skipped (want 6 fields): %r", line)
            continue
        try:
            major, _, minor = parts[5].partition(".")
            gpus.append(GpuInfo(
                index=int(parts[0]),
                uuid=parts[1],
                name=parts[2],
                mem_total_mib=int(float(parts[3])),
                mem_used_mib=int(float(parts[4])),
                compute_cap=(int(major), int(minor or 0)),
            ))
        except (ValueError, IndexError) as exc:
            log.warning("nvidia-smi row skipped (%s): %r", exc, line)
    return gpus


def _probe_via_smi(timeout_s: float) -> Optional[List[GpuInfo]]:
    try:
        out = subprocess.run(
            _SMI_CMD, capture_output=True, text=True, timeout=timeout_s, check=False,
        )
    except FileNotFoundError:
        return None  # no nvidia-smi on PATH → try the torch child
    except subprocess.TimeoutExpired:
        log.error("nvidia-smi hung >%.0fs (wedged driver?) — treating as no GPUs", timeout_s)
        return []
    if out.returncode != 0:
        log.warning("nvidia-smi rc=%s: %s", out.returncode, (out.stderr or "").strip()[:200])
        return []
    return parse_smi_csv(out.stdout)


# child probe: torch CUDA context lives and dies with the child process
_TORCH_PROBE_SRC = r"""
import json, torch
gpus = []
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        free, total = torch.cuda.mem_get_info(i)
        gpus.append({"index": i, "uuid": "", "name": p.name,
                     "mem_total_mib": total // 2**20,
                     "mem_used_mib": (total - free) // 2**20,
                     "cc": [p.major, p.minor]})
print(json.dumps(gpus))
"""


def _probe_via_torch_child(timeout_s: float) -> List[GpuInfo]:
    try:
        out = subprocess.run(
            [sys.executable, "-c", _TORCH_PROBE_SRC],
            capture_output=True, text=True, timeout=timeout_s, check=False,
        )
        rows = json.loads(out.stdout.strip() or "[]") if out.returncode == 0 else []
    except (subprocess.TimeoutExpired, json.JSONDecodeError, OSError) as exc:
        log.warning("torch child probe failed: %s", exc)
        return []
    return [
        GpuInfo(index=r["index"], uuid=r.get("uuid", ""), name=r.get("name", "?"),
                mem_total_mib=int(r["mem_total_mib"]), mem_used_mib=int(r["mem_used_mib"]),
                compute_cap=tuple(r.get("cc", (0, 0))))
        for r in rows
    ]


def probe_topology(timeout_s: float = 10.0) -> List[GpuInfo]:
    """Detect the box's GPUs. [] = no usable GPUs (CPU box or wedged driver)."""
    gpus = _probe_via_smi(timeout_s)
    if gpus is None:  # nvidia-smi absent — a torch child probe is the fallback
        gpus = _probe_via_torch_child(timeout_s * 6)  # cold torch import is slow
    log.info("GPU topology: %s", [f"{g.index}:{g.name}({g.mem_free_mib}MiB free,cc{g.compute_cap[0]}.{g.compute_cap[1]})" for g in gpus] or "none")
    return gpus


def select_attn_impl(compute_cap: Tuple[int, int], override: str = "auto") -> str:
    """Resolve the attention implementation for a GPU.

    Explicit override wins; `auto` avoids flash-attn only where it categorically
    cannot run (no cubins for the arch — Blackwell sm_120+). A wrong-arch FA
    build below that (e.g. an sm_90-only wheel on the 4090) imports and loads
    fine but fails at runtime, which the worker's warmup forward catches.
    """
    override = (override or "auto").strip().lower()
    if override and override != "auto":
        return override
    if tuple(compute_cap) >= FLASH_ATTN_UNSUPPORTED_CC:
        return "sdpa"
    return "flash_attention_2"
