# VL Realtime 独立部署（调用端到模型的实时视觉协议 §5）

面向视觉调用端（`VISION_RUNTIME_URL` / `VL_MODEL_WS_URL`）的 sglang 独立部署：
**只提供 VL realtime WebSocket 面**，不含 demo 前端、ASR、TTS。推理后端为
sglang-omni realtime server（`/v1/video/realtime`），本服务在其前面做 §5 协议
适配、副本池与容量管理。

## 部署

```bash
SGLANG_OMNI_URLS=http://127.0.0.1:18500 scripts/deploy/run_vision.sh
# 或者直接：
<venv>/python -m uvicorn server.gateway.vision:app --host 0.0.0.0 --port 8010 \
    --ws-max-size 67108864 --ws-ping-interval 20 --ws-ping-timeout 20
```

- 端点：`ws://<host>:<port>/v1/realtime?session_id=`（`session_id` 可为空，接受
  但忽略；不要求任何鉴权头；网关前缀由部署入口的 root_path/反代决定）。
- 无副本可达时 lifespan 启动即失败（fail-fast），不会带着坏池子起服务。
- `GET /v1/realtime/health`：副本状态与 slot 水位（运维面）。

## 协议实现（对应交接文档 §5.1–§5.5）

每轮分析一个 WebSocket 连接：`start` → `ready` → N×(`frame`+二进制 JPEG) →
N×`frame_ack` → `output` 增量（以 `<|im_end|>` 结束）→ `stop` → 关闭。

| 调用端约定 | 本服务行为 |
|---|---|
| `start`（§5.2） | 解析并映射到 omni `session.configure`；未知字段忽略，绝不因字段/组合拒绝正常请求 |
| `ready` | omni `session.ready` 后返回 `{"type":"ready"}` |
| `frame` + 二进制 JPEG（§5.3） | 元数据下一条必须是二进制体；两阶段转发；接纳成功即回 `frame_ack`；支持批量到达（先全发后收 ack） |
| `output`（§5.4） | omni delta 原样透传（剥离 `<|round_start|>` 控制 token）；`response.turn.silence` → 追加 `{"type":"output","text":"<|im_end|>"}`；仅当已有非空可见文字才发结束标记 |
| `stop`（§5.4） | `session.abort` 释放会话与 slot，正常关闭（1000）；客户端不等确认直接断开也能清理 |
| 忙（§5.5） | 无空闲 slot → `{"type":"error","message":"realtime session is already active"}`（含调用端重试识别串），连接随即关闭 |
| 错误（§5.5） | `{"type":"error","message":"<可读原因>"}`（无 code 字段），随后关闭并清理 slot，下一轮可立即再调用 |
| 断连清理（§5.5） | stop/错误/断连均 abort 后端会话并归还 slot；不发送 `session_end`；完成后不抢先断开 |

### start 参数映射

| start 字段 | omni configure 字段 | 说明 |
|---|---|---|
| `prompt` | `prompt` | 必填非空，否则 error（唯一硬拒绝项） |
| `max_new_tokens` | `max_new_tokens` | 钳制 1..`VISION_MAX_NEW_TOKENS`(默认 2048) |
| `max_tokens_per_second` | `max_tokens_per_turn` | omni 即 tokens/second 速率上限，语义一致 |
| `do_sample=false` | `temperature=0.0` | 贪心；`temperature` 字段此时被忽略 |
| `temperature` / `top_p` | `temperature` / `top_p` | 仅 do_sample=true 时生效 temperature |
| `frame_queue_size` | `input_queue_capacity` | 钳制 1..`VISION_MAX_INPUT_QUEUE`(默认 32)；保证一批帧全部在途、不触发背压丢帧 |
| `top_k` / `repetition_penalty` | —（忽略） | omni configure 不支持；记录日志一次，不影响请求 |

有意忽略项即上表"忽略"行与未知字段；不支持的采样能力（top_k、repetition_penalty、
do_sample 的采样开关本身）属交付口径，见下方"边界与已识别限制"。

## 配置（均可用环境变量覆盖，见 .env.deploy.example）

- `SGLANG_OMNI_URLS`：sglang-omni 副本列表（逗号分隔），模型选择由部署地址决定。
- `SGLANG_OMNI_SESSIONS_PER_REPLICA`：每副本并发 slot（须等于 omni 侧
  `--max-running-requests`）；全部占满时对调用端表现为"忙"。
- `GATEWAY_MAX_FRAME_BYTES`：单帧上限（默认 32 MiB ≥ 调用端 10 MiB）；
  `run_vision.sh` 自动把传输层 `--ws-max-size` 设为其 2 倍。
- `VISION_START_TIMEOUT_S` / `VISION_FRAME_TIMEOUT_S` / `VISION_ROUND_TIMEOUT_S`：
  start 读取 / 帧元数据→二进制间隔 / 整轮兜底（默认 10 / 10 / 120 秒）。

## 测试与验证

`server/tests/test_gateway_vision.py`（7 用例，fake omni 后端）覆盖：完整轮次
（批量帧→逐帧 ack→控制 token 剥离→`<|im_end|>`）、忙重试串与恢复、错误图片、
中途断连清理与再调用、宽松 start 契约、独立应用 fail-fast 与端到端。回归：
`test_gateway_rest/ws/lifecycle/metrics/qa`、`test_sglang_omni_adapter`、
`test_config_layering` 等全部通过（详见提交说明）。

## 边界与已识别限制（交接口径）

1. **时延预算**：调用端单轮总时限 10s（含握手与重试）、ready/帧确认等待 10s、
   输出静默 1s。服务端开销（适配层）为毫秒级；端到端首段/完整时延由
   sglang-omni 实测交付，超出 10s 需按文档提出配置调整。
2. **时间戳单调**：后端要求传输时间戳单调；乱序到达帧的时间戳被钳制为不小于
   前一帧（不影响帧内容与顺序）。
3. **忙的语义**：调用端只重试忙；本服务把"无空闲 slot"与"全部副本不可达"
   都映射为忙串（副本探活恢复通常在秒级，重试窗口内可自愈）。
4. **帧大小**：调用端单帧 ≤10 MiB < 服务端 32 MiB 上限；超限帧返回 error 并
   关闭（close 1009 口径）。
5. **只支持 JPEG/PNG/WebP**：其余编码按错误图片路径处理（§5.5 要求支持）。
6. **每轮一连接**：一轮结束后如需继续分析，调用端重新建连即可；服务端不保留
   任何已结束的忙状态。
7. **合流路径**：本文件组（vision.py / run_vision.sh / test_gateway_vision.py /
   pool.py 的 input_queue_capacity 覆盖 / config 的 vision_* 字段）可干净落在
   `npu/full → main` 合流后的主干上——omni adapter 与 gateway 平面代码在
   origin/main 已存在且与本分支一致。
