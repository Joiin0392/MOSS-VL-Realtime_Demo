#!/usr/bin/env bash
# stop_demo.sh — 一键关停本框架全栈（start_demo.sh 的反向操作，同在 GPU 节点执行）。
#
# 关停范围（都是本框架自己的端口，与他人服务无关）：
#   gateway/web/TTS sidecar (demo.sh down) + omni 实例
#   + memory 整套（pi_agent:38082 + 4B decide/compact :38090——与本框架强绑定，同起同停）
#   + 两条转发（-R 20941 入口 / -D 17890 MiniMax 出口）
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CPU_PORT=${CPU_PORT:-20941}
MM_SOCKS_PORT=${MM_SOCKS_PORT:-17890}

echo "==> [1/4] gateway + TTS + web"
(cd "$REPO" && bash scripts/deploy/demo.sh down 2>&1 | grep -v "rtunnel\|fkqz\|cloudflared\|NOTE" || true)

echo "==> [2/4] sglang-omni 实例"
killed=0
for pidfile in "$REPO"/logs/sglang_omni/omni_*_p*.pid; do
  [ -f "$pidfile" ] || continue
  pid=$(cat "$pidfile" 2>/dev/null || true)
  if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then kill "$pid" 2>/dev/null && killed=1; fi
  rm -f "$pidfile"
done
[ "$killed" = "1" ] && echo "      omni 实例已停" || echo "      无运行中的 omni 实例"

echo "==> [3/4] memory 整套（pi_agent :38082 + 4B :38090）"
pid=$(cat "$REPO/logs/pi_agent/pi_agent_38082.pid" 2>/dev/null || true)
if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then kill "$pid" && echo "      pi_agent 已停 (pid $pid)"; else echo "      无运行中的 pi_agent"; fi
# [l]aunch_server 写法防自匹配；只认 38090 端口特征，不误伤别人的 sglang
p4b="$(pgrep -f '[l]aunch_server.*--port 38090' || true)"
if [ -n "$p4b" ]; then kill $p4b 2>/dev/null && echo "      4B :38090 已停 (pid $p4b)"; else echo "      4B :38090 未运行"; fi

echo "==> [4/4] 转发"
for spec in "-R 127.0.0.1:$CPU_PORT:" "-D 127.0.0.1:$MM_SOCKS_PORT"; do
  pids="$(ps -ww -eo pid=,comm=,args= | awk -v s="$spec" '$2 == "ssh" && index($0, s) {print $1}')"
  for pid in $pids; do kill "$pid" 2>/dev/null && echo "      清掉转发 pid $pid ($spec)" || true; done
done

echo
echo "已全部关停。显存几秒后回落到 holder 基线（~730MiB/卡）。"
