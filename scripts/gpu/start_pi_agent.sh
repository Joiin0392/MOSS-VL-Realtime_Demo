#!/usr/bin/env bash
# start_pi_agent.sh — 在 GPU 节点拉起 demo 专用的 pi_agent（memory decide/compact）。
#
# 从 CPU 节点直接执行即可（自动经反向隧道 ssh 到 GPU 节点）。
# 本框架 pi_agent 默认端口 PI_PORT=38082。
# decide/compact 后端默认复用 GPU 节点已运行的 Qwen3-4B sglang（127.0.0.1:38090）；
# 若它没在跑且 START_4B=1，本脚本会先拉起 4B（占 DECIDE_LLM_GPU，默认 0 号卡）。
set -euo pipefail

GPU_SSH=${GPU_SSH:-"ssh -p 10008 -o BatchMode=yes -o ConnectTimeout=10 root@127.0.0.1"}

if [ "${_PI_ON_GPU:-0}" != "1" ]; then
  exec $GPU_SSH "_PI_ON_GPU=1 \
    OMNI_ROOT='${OMNI_ROOT:-}' PI_AGENT_NODE='${PI_AGENT_NODE:-}' \
    PI_PORT='${PI_PORT:-}' START_4B='${START_4B:-}' \
    DECIDE_LLM_MODEL='${DECIDE_LLM_MODEL:-}' DECIDE_LLM_GPU='${DECIDE_LLM_GPU:-}' \
    LOG_DIR='${LOG_DIR:-}' bash -s" < "$0"
fi

# ---------------- 以下在 GPU 节点执行 ----------------
OMNI_ROOT=${OMNI_ROOT:-/inspire/qb-ilm/project/video-understanding/public/train/moss_vl_streaming/8B/final_release/MOSS-VL-Realtime-sglang}
PI_AGENT_DIR="$OMNI_ROOT/pi_agent"
PI_AGENT_NODE=${PI_AGENT_NODE:-/inspire/hdd/project/video-understanding/public/personal/yxchen/.local/node-v22.12.0-linux-x64/bin/node}
PI_PORT=${PI_PORT:-38082}
START_4B=${START_4B:-0}
DECIDE_LLM_MODEL=${DECIDE_LLM_MODEL:-/inspire/hdd/project/video-understanding/public/share/models/Qwen3-4B-Instruct-2507}
DECIDE_LLM_GPU=${DECIDE_LLM_GPU:-0}
LOG_DIR=${LOG_DIR:-/inspire/hdd/project/video-understanding/public/personal/yxchen/mossvl_realtime_inference/logs/pi_agent}
mkdir -p "$LOG_DIR"

# 4B decide/compact 后端：默认复用已运行的 38090；没有且 START_4B=1 才拉起
# DECIDE_LLM_MEM_FRAC 默认 0.2：omni 实例同卡占 0.6 时，0.08 已不够权重+KV
# （实测最低要求 0.146），0.2 (~29GB) 在同卡共存场景下安全
if curl -s --max-time 3 http://127.0.0.1:38090/health >/dev/null 2>&1; then
  echo "4B decide/compact backend already healthy on :38090, reusing."
elif [ "$START_4B" = "1" ]; then
  echo "Starting 4B backend on :38090 (GPU $DECIDE_LLM_GPU, mem-frac ${DECIDE_LLM_MEM_FRAC:-0.2})..."
  (cd "$OMNI_ROOT/sglang-omni-main" && \
   PATH="$OMNI_ROOT/.venv-main/bin:$PATH" \
   CUDA_VISIBLE_DEVICES="$DECIDE_LLM_GPU" nohup "$OMNI_ROOT/.venv-main/bin/python" \
     -m sglang.launch_server --model-path "$DECIDE_LLM_MODEL" \
     --host 127.0.0.1 --port 38090 --mem-fraction-static "${DECIDE_LLM_MEM_FRAC:-0.2}" --disable-radix-cache \
     > "$LOG_DIR/decide_llm.log" 2>&1 &)
else
  echo "WARNING: 4B backend :38090 not healthy and START_4B!=1; pi_agent 会启动但 decide/compact 会失败" >&2
fi

# 本地 4B 配置（横评结论：JSON/逐字率打平网关大模型，延迟 1.5s vs 10s+）
export AIGW_DECIDE_BASE_URL=${AIGW_DECIDE_BASE_URL:-http://127.0.0.1:38090/v1}
export AIGW_DECIDE_MODEL=${AIGW_DECIDE_MODEL:-Qwen3-4B-Instruct-2507}
export AIGW_COMPACT_BASE_URL=${AIGW_COMPACT_BASE_URL:-http://127.0.0.1:38090/v1}
export AIGW_COMPACT_MODEL=${AIGW_COMPACT_MODEL:-Qwen3-4B-Instruct-2507}
export PI_COMPACT_TIMEOUT_MS=${PI_COMPACT_TIMEOUT_MS:-30000}
export PI_PORT

# 只杀本端口旧实例（pidfile 优先）
old=""
[ -f "$LOG_DIR/pi_agent_${PI_PORT}.pid" ] && old=$(cat "$LOG_DIR/pi_agent_${PI_PORT}.pid" 2>/dev/null || true)
{ [ -z "$old" ] || ! kill -0 "$old" 2>/dev/null; } && old=$(pgrep -f "node service.mjs" -a 2>/dev/null | while read -r pid _; do
    tr '\0' '\n' < "/proc/$pid/environ" 2>/dev/null | grep -q "^PI_PORT=$PI_PORT$" && echo "$pid"
  done | head -1 || true)
if [ -n "$old" ]; then
  echo "Restarting demo pi_agent on :$PI_PORT (killing PID: $old)"
  kill $old 2>/dev/null || true
  sleep 2
fi

[ -x "$PI_AGENT_NODE" ] || { echo "FATAL: node not found: $PI_AGENT_NODE" >&2; exit 1; }
[ -f "$PI_AGENT_DIR/service.mjs" ] || { echo "FATAL: $PI_AGENT_DIR/service.mjs not found" >&2; exit 1; }

cd "$PI_AGENT_DIR"
nohup "$PI_AGENT_NODE" service.mjs > "$LOG_DIR/pi_agent_${PI_PORT}.log" 2>&1 &
echo $! > "$LOG_DIR/pi_agent_${PI_PORT}.pid"
for _ in $(seq 1 15); do
  if curl -s --max-time 2 "http://127.0.0.1:$PI_PORT/health" >/dev/null 2>&1; then
    echo "pi_agent ready on :$PI_PORT"
    curl -s --max-time 2 "http://127.0.0.1:$PI_PORT/health"; echo
    exit 0
  fi
  sleep 1
done
echo "WARNING: pi_agent did not become healthy in 15s" >&2
exit 1
