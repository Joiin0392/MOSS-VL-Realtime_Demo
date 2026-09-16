#!/usr/bin/env bash
# Standalone VL realtime service (调用端到模型的实时视觉协议 §5).
#
# One process, one endpoint: WS /v1/realtime — no demo plane, no ASR/TTS, no
# frontend. Requires SGLANG_OMNI_URLS pointing at running sglang-omni realtime
# servers (see .env.deploy.sglang-omni.example for the full env surface).
#
# Usage:
#   SGLANG_OMNI_URLS=http://127.0.0.1:18500 scripts/deploy/run_vision.sh
# Environment: PORT (default 8010), HOST, GATEWAY_MAX_FRAME_BYTES,
#   WS_MAX_SIZE, WS_PING_INTERVAL/WS_PING_TIMEOUT, LOG_LEVEL.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"

if [[ -z "${SGLANG_OMNI_URLS:-}" ]]; then
  echo "Error: SGLANG_OMNI_URLS is required (e.g. http://127.0.0.1:18500)" >&2
  exit 1
fi

# Leave room above the application frame limit so an oversize frame gets an
# error event before close 1009 (same discipline as run_backend.sh).
frame_limit=${GATEWAY_MAX_FRAME_BYTES:-33554432}
[[ "$frame_limit" =~ ^[1-9][0-9]*$ ]] || { echo "Invalid GATEWAY_MAX_FRAME_BYTES" >&2; exit 1; }
ws_limit=${WS_MAX_SIZE:-$((frame_limit * 2))}
[[ "$ws_limit" =~ ^[1-9][0-9]*$ ]] || { echo "Invalid WS_MAX_SIZE" >&2; exit 1; }
(( ws_limit > frame_limit )) || { echo "WS_MAX_SIZE must exceed GATEWAY_MAX_FRAME_BYTES" >&2; exit 1; }

PYBIN="${PYBIN:-$REPO/.venv/bin/python}"
[ -x "$PYBIN" ] || PYBIN="$(command -v python3 || echo python3)"

export MOSS_LOG_FILE="${MOSS_LOG_FILE:-$REPO/logs/handler/backend/vision.log}"
export PYTHONUNBUFFERED=1

exec "$PYBIN" -m uvicorn server.gateway.vision:app \
  --host "${HOST:-0.0.0.0}" --port "${PORT:-8010}" \
  --loop "${UVICORN_LOOP:-uvloop}" \
  --ws-max-size "$ws_limit" \
  --ws-ping-interval "${WS_PING_INTERVAL:-20}" --ws-ping-timeout "${WS_PING_TIMEOUT:-20}"
