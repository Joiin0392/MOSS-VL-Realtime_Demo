# MOSS-VL-Realtime Demo 调用链分析

## 完整调用链

```
Browser (WebSocket mic/video/events)
  │
  └─ server/routers/session_ws.py
       └─ server/session/orchestrator.py (5个异步循环)
            │
            ├─ [1] audio_loop ─────────── 接收 PCM → VAD → ASR → 转写文本 → VLM prompt
            │
            ├─ [2] vlm_drain_loop ─────── 轮询VLM输出队列 → 路由控制token + 内容文本
            │    │                             内容文本 → Segmenter切句 → TTS unit入队
            │    │                             控制token → 轮次管理 (round_start/end/silence)
            │
            ├─ [3] tts_feeder_loop ────── 读取TTS unit队列 → 背压/丢过期 → feed_segment()
            │    │                             → TTS引擎合成 → PCM回调 → _tts_events队列
            │
            ├─ [4] tts_pump_loop ──────── 读取 _tts_events 队列 → emit到客户端 WS
            │    │                             (response.audio.delta + binary PCM)
            │
            └─ [5] status_loop ────────── 1Hz状态心跳 → 触发rollover检查
```

## 离线聊天调用链（chat 页，图文问答）

与实时会话（realtime WS）完全独立，走 REST/WS chat 通道，不经过 orchestrator 的 5 个循环：

```
聊天页 (Browser)
  │
  ├─ WS  /api/chat/stream   (chat_stream_ws)
  └─ SSE POST /api/chat/stream  (chat_stream_sse)
       │
       ▼
  routers/chat.py:_chat_vlm()   ← 后端选择（chat.py:36-47）
       │
       ├─ 首选：vlm_offline (MOSS-VL-Instruct-0708 @ sglang, NPU卡7/15)
       │    └─ SglangOfflinePool.generate_stream()
       │         ├─ 统一路径：apply_chat_template(消息+<|image|>/<|video|>占位)
       │         └─ HTTP POST → sglang sidecar /generate (stream=true)
       │              ├─ 纯文本：body只有 text+sampling_params
       │              └─ 图文  ：body附加 image_data / video_data
       │              → SSE → 文本delta
       │
       └─ 回退：rt.vlm (MOSS-VL-Realtime-0708 @ 在线worker池)
            └─ HfMossVlAdapter.generate_stream()  (offline模式decode loop)
                 ├─ 图文  → processor → vision prefill → decode（显式分支）
                 └─ 纯文本 → tokenizer模板 → decode
              （sglang不可用时降级，1-GPU盒/未构建sidecar场景）
```

**两条后端对纯文本/图文的分工差异**：
- **sglang 路径**：一个统一请求流，媒体只是可选字段（有图才附加 `image_data`/`video_data`）
- **HF 回退路径**：代码里显式 `if images or videos:` 分支（adapter.py:723-748），图文走 processor + vision prefill，纯文本走 tokenizer 模板

**注意**：`_chat_vlm()` 会先查 sglang 离线面的健康状态（`is_loaded()`），挂了才回退在线池——所以纯文本/图文聊天默认走 **MOSS-VL-Instruct-0708**，回退时才用 MOSS-VL-Realtime-0708。

### 5个循环的数据传递关系（含模型）

```
                    ┌───────────────────────────────────────────────────────────┐
                    │                        Orchestrator                        │
                    │                                                            │
 客户端WS ──16kHz PCM──→ [audio_loop]                                            │
                    │      │                                                    │
                    │      ▼                                                    │
                    │  ┌───────────────────────────────────────────────────┐   │
                    │  │   ASR 模型：SenseVoiceSmall + fsmn-vad            │   │
                    │  │   [FunASR 1.3.1 · NPU卡4]                         │   │
                    │  │   Input: 16kHz PCM → Output: 转写文本             │   │
                    │  └───────────────────────────────────────────────────┘   │
                    │      │ 转写文本                                          │
                    │      ▼ _user_turn()                                      │
                    │  put_prompt / put_prompt_frame                           │
                    │      │                                                  │
                    │      ▼                                                  │
                    │  ┌───────────────────────────────────────────────────┐   │
                    │  │  VLM 模型：MOSS-VL-Realtime-0708                  │   │
                    │  │  [HF Transformers 4.57 · NPU卡0]                  │   │
                    │  │  Input: JPEG帧 + text prompt                      │   │
                    │  │  Output: text token / 控制token                   │   │
                    │  │  real_time_generate() → output_queue              │   │
                    │  └───────────────────────────────────────────────────┘   │
                    │      │ poll_output                                      │
                    │      ▼                                                  │
                    │  [vlm_drain_loop]                                       │
                    │      ├─ 控制token → _open_response / _finalize_response │
                    │      └─ 内容文本 → Segmenter → _push_unit()             │
                    │                                   │                     │
                    │                                   ▼                     │
                    │                            _units deque                  │
                    │                                   │                     │
                    │  [tts_feeder_loop] ←──── pop unit ─┘                    │
                    │      ├─ 背压控制 (audio_queue_seconds /                 │
                    │      │    should_emit_next_unit)                        │
                    │      ├─ 丢过期 (drop_stale)                             │
                    │      └─ feed_segment()                                  │
                    │           │                                             │
                    │           ▼                                             │
                    │  ┌───────────────────────────────────────────────────┐   │
                    │  │  TTS 模型：MOSS-TTS-Nano-100M +                    │   │
                    │  │           MOSS-Audio-Tokenizer-Nano                │   │
                    │  │  [PyTorch + torchair · NPU卡1 · HTTP sidecar]     │   │
                    │  │  Input: text + 音色参考 → Output: PCM16LE chunk    │   │
                    │  │  MossTtsNanoAdapter → nano_protocol →             │   │
                    │  │  sidecar: NanoTTSService.synthesize_stream()      │   │
                    │  │    ├─ MOSS-TTS-Nano-100M (text → audio token)     │   │
                    │  │    └─ Audio-Tokenizer-Nano (token → PCM)          │   │
                    │  └───────────────────────────────────────────────────┘   │
                    │      │ _tts_emit_threadsafe()                           │
                    │      ▼                                                  │
                    │  _tts_events Queue                                      │
                    │      │                                                  │
                    │  [tts_pump_loop] ←──── get event ─┘                     │
                    │      │                                                  │
                    │      ▼                                                  │
                    │  state.emit(response.audio.delta + binary PCM)          │
                    │      │                                                  │
                    │  [status_loop] ──1Hz──→ status事件 → 客户端WS           │
                    └──────┬───────────────────────────────────────────────────┘
                           │
                           ▼
                    客户端WS → 浏览器播放
```

### 各循环详解

| 循环 | 函数 | 消费什么 | 产出什么 | 传递方式 |
|------|------|----------|----------|----------|
| audio_loop | `_audio_loop` | `_audio_in` deque（PCM + 控制标记） | ASR转写文本 → VLM prompt | `asyncio.to_thread(stream.send_pcm)` → `stream.finalize()` → `_user_turn()` |
| vlm_drain_loop | `_vlm_drain_loop` | `VlmRealtimeSession.output_queue`（模型输出token） | 控制token（轮次管理）+ 内容文本（TTS unit） | `poll_output()` → `_route_model_text()` → `_push_unit()` 塞入 `_units` deque |
| tts_feeder_loop | `_tts_feeder_loop` | `_units` deque（TtsUnit） | 文本分段 → TTS引擎 → PCM chunk | `feed_segment()` → TTS worker线程 → `_tts_emit_threadsafe()` → `_tts_events` Queue |
| tts_pump_loop | `_tts_pump_loop` | `_tts_events` asyncio.Queue | `response.audio.delta` + binary PCM → 客户端 | `state.emit(p.RESPONSE_AUDIO_DELTA, binary=...)` |
| status_loop | `_status_loop` | VLM状态 + 队列深度 | `session.status` 事件 → 客户端 | `state.emit(p.STATUS, ...)` 1Hz |

### 为什么叫"VLM排水循环"和"TTS喂料循环"

- **VLM排水循环**（vlm_drain_loop）：VLM 模型在后台线程中持续生成 token，输出堆积在 `output_queue` 中。这个循环负责不断"排水"——轮询 `poll_output()` 把模型产出的 token 捞出来，分流到控制逻辑和 TTS。相当于从 VLM 的输出端持续抽水。

- **TTS喂料循环**（tts_feeder_loop）：VLM 产出的文本经过 Segmenter 切句后，变成一个个 TTS unit 塞入 `_units` deque。这个循环负责从 deque 中取出 unit，经过背压判断（是否应该继续合成）和过期丢弃后，"喂"给 TTS 引擎合成语音。相当于给 TTS 引擎持续投料。```

---

## 已适配模型详情

### 1. MOSS-VL-Realtime-0708（在线流式VLM）

| 项目 | 内容 |
|------|------|
| **角色** | 实时音视频会话（在线流式） |
| **输入** | JPEG帧（WebCam） + 文本prompt |
| **输出** | 文本token + 控制token（`<\|silence\|>`, `<\|round_start\|>`, `<\|eot_id\|>` 等） |
| **推理框架** | HF Transformers 4.57，自定义 `MossVLForConditionalGeneration`（remote code） |
| **Attention** | NPU: `eager`（CANN 9.0 SDPA 在 3-D M-RoPE 解码时 hang）；CUDA: `flash_attention_2` / `sdpa` |
| **部署** | NPU 卡0（`VLM_DEPLOY=inproc` 网关进程内）或独立 worker 进程（`VLM_DEPLOY=workers`） |
| **关键代码** | `server/adapters/vlm/moss_vl_hf/adapter.py` + `serving_policy.py` |
| **NPU 适配** | GIL 忙等修复（52.5s→150ms）、`empty_cache` NPU 感知、10D permute / BICUBIC 链、`ATTN_IMPL=eager` |
| **权重来源** | 内部分发 |

### 2. MOSS-VL-Instruct-0708（sglang 离线面）

| 项目 | 内容 |
|------|------|
| **角色** | 离线图文问答（多轮对话） |
| **输入** | Chat messages + 图片（base64 / CAS handle） + 视频（CAS handle） |
| **输出** | 流式文本 delta（SSE） |
| **推理框架** | sglang fork（Joiin0392/MOSS-VL 仓库），`sgl-kernel-npu` + `triton-ascend 3.2.0` |
| **Attention** | sglang ascend backend |
| **部署** | NPU 卡7/15（`OFFLINE_GPU_RATIO=0.25` 自动比例或 `OFFLINE_GPUS` 显式钉卡） |
| **关键代码** | `server/adapters/vlm/moss_vl_sglang/adapter.py` → `POST /generate` HTTP |
| **NPU 适配** | 13 commits：npu_utils 设备抽象、ascend backend 5 个稳定性修复、torch-npu 2.10 兼容 |
| **权重来源** | HF OpenMOSS-Team |

### 3. SenseVoiceSmall + fsmn-vad（ASR 组合）

| 项目 | 内容 |
|------|------|
| **角色** | 语音识别（VAD 切分 + 转写） |
| **输入** | 16kHz PCM 音频（WAV 临时文件） |
| **输出** | 转写文本（ITN 后处理） |
| **推理框架** | FunASR 1.3.1（ModelScope），`funasr.AutoModel` |
| **Attention** | 非 LLM，无 attention 选择 |
| **部署** | NPU 卡4（独立隔离；原与卡0共卡，GIL 问题后迁出） |
| **关键代码** | `server/adapters/asr/funasr_sensevoice/adapter.py` |
| **NPU 适配** | Demo 侧零修改，funasr 原生接受 `device="npu:4"` |
| **权重来源** | ModelScope iic/* |

### 4. MOSS-TTS-Nano-100M + MOSS-Audio-Tokenizer-Nano（TTS 组合）

| 项目 | 内容 |
|------|------|
| **角色** | 本地 TTS（文本→音频 token + token→PCM） |
| **输入** | 文本 + 音色名（prompt 音频参考文件） |
| **输出** | PCM16LE 流式音频 chunk（48kHz, 2ch） |
| **推理框架** | PyTorch（主） / ONNX Runtime（备选），两阶段流水线 |
| **Attention** | NPU: `sdpa` / math path（`masked_fill`）；CUDA: `flash_attention_2` |
| **部署** | NPU 卡1，独立 sidecar 进程（FastAPI HTTP），`torchair` 图编译 |
| **关键代码** | `server/adapters/tts/moss_tts_nano/` + sidecar `moss_tts_nano_sidecar.py` + `moss_tts_nano_runtime.py` |
| **NPU 适配** | torchair 编译钩子（`MOSS_TTS_NANO_TORCHAIR=1`）、prompt-codes 按音色缓存（127→50ms）、pad-to-bucket KV（320/640 桶） |
| **权重来源** | ModelScope OpenMOSS/* |

---

## 推理框架总览（带版本号）

### NPU 运行环境（统一环境 `moss_tts_210`）

| 项 | 版本 |
|----|------|
| 硬件 | Ascend 910B2C ×8（CANN 9.0.0） |
| Python | 3.11 |
| torch / torchvision / torchaudio | 2.10.0+cpu / 0.25.0+cpu / 2.10.0+cpu |
| torch-npu | 2.10.0.post4（内含 torchair） |
| transformers | 4.57.6（主线；5.x 兼容层为 5.12.1） |
| 部署方式 | `server/device_compat.py` 设备抽象层统一调度 |

### 各模型推理框架

| 模型 | 推理框架（版本） | Attention | 图编译 / 加速 |
|------|------------------|-----------|--------------|
| MOSS-VL-Realtime-0708（在线VLM） | HF Transformers **4.57.6** + torch **2.10.0+cpu** + torch-npu **2.10.0.post4** | `eager`（NPU 无 flash-attn；CANN 9.0 SDPA 在 3-D M-RoPE 解码时 hang） | ❌ 无图编译 |
| MOSS-VL-Instruct-0708（离线sglang） | sglang fork（Joiin0392/MOSS-VL `npu/torch210-compat` 分支，source 安装）+ torch **2.10.0+cpu** + torch-npu **2.10.0.post4** | sglang ascend backend | `sgl-kernel-npu` **2026.8.10**（torch2.10 轮子）+ `triton-ascend` **3.2.0**（CANN 9.0 兼容补丁） |
| SenseVoiceSmall + fsmn-vad（ASR） | FunASR **1.3.1**（ModelScope），`funasr.AutoModel` + pynini **2.1.7** + WeTextProcessing **1.2.0** | N/A（非 LLM） | ❌ 无图编译 |
| MOSS-TTS-Nano-100M + MOSS-Audio-Tokenizer-Nano（TTS） | PyTorch **2.10.0+cpu** + torch-npu **2.10.0.post4**（主）；ONNX Runtime **1.23.2**（备选） | `sdpa` / math path（`masked_fill`，torchair 无 npu_fusion_attention_v3 converter 时） | `torchair`（内置于 torch-npu）图编译 |

### GPU 参考环境（requirements.txt，H200 板）

| 项 | 版本 |
|----|------|
| torch / torchvision / torchaudio / torchcodec | 2.8.0 / 0.23.0 / 2.8.0 / 0.7.0 |
| transformers | 4.57.1 |
| funasr | 1.3.14 |
| flash-attn | 2.8.1（cu12/torch2.8 构建） |
| sglang fork（GPU 侧） | v0.5.5.post3 + flashinfer 0.5.2 + sgl-kernel 0.3.17.post1 |

---

## NPU 卡布局（Ascend 910B2C ×8）

| 卡号 | 模型 | 部署方式 |
|------|------|----------|
| 卡0 | MOSS-VL-Realtime-0708 | inproc（网关进程内） |
| 卡1 | MOSS-TTS-Nano-100M + MOSS-Audio-Tokenizer-Nano | torchair 图编译 sidecar |
| 卡4 | SenseVoiceSmall + fsmn-vad | 独立 sidecar（隔离） |
| 卡7/15 | MOSS-VL-Instruct-0708 | sglang sidecar |

---

## 数据流路径（5个循环串联）

```
                          ┌──────────────────────────────────────────────┐
                          │             Orchestrator                     │
                          │                                              │
用户说话 ──16kHz PCM──→   │  [audio_loop]                               │
                          │    ├─ VAD (RMS端点检测, auto/PTT模式)        │
                          │    └─ ASR (SenseVoiceSmall + fsmn-vad)       │
                          │         → 转写文本                           │
                          │         → _user_turn() → VLM prompt          │
                          │                                              │
用户文本 / ASR文本        │  → VLM (MOSS-VL-Realtime-0708)              │
  + 摄像头JPEG帧          │    → real_time_generate() 循环               │
                          │    → output_queue (token流)                  │
                          │                                              │
                          │  [vlm_drain_loop]  ← poll_output            │
                          │    ├─ 控制token                              │
                          │    │  ├─ <|round_start|> → _open_response   │
                          │    │  ├─ <|silence|> → _finalize_response   │
                          │    │  ├─ <|eot_id|> → _finalize_response   │
                          │    │  └─ <|response|> → 隐藏                │
                          │    └─ 内容文本                               │
                          │         → 字幕delta (response.text.delta)    │
                          │         → Segmenter切句 → _push_unit()      │
                          │         → _units deque                       │
                          │                                              │
                          │  [tts_feeder_loop]  ← pop unit              │
                          │    ├─ 背压控制 (audio_queue_seconds)         │
                          │    ├─ 丢过期 (drop_stale, tts_max_pending)   │
                          │    ├─ backlog coalescing (合并相邻unit)      │
                          │    └─ feed_segment() → TTS引擎              │
                          │         → MossTtsNanoAdapter                  │
                          │         → HTTP sidecar                        │
                          │         → NanoTTSService.synthesize_stream()  │
                          │           ├─ MOSS-TTS-Nano-100M (text→token) │
                          │           └─ MOSS-Audio-Tokenizer-Nano       │
                          │                (token→PCM)                   │
                          │         → _tts_emit_threadsafe()             │
                          │         → _tts_events Queue                  │
                          │                                              │
                          │  [tts_pump_loop]  ← _tts_events             │
                          │    └─ state.emit(response.audio.delta)       │
                          │         + binary PCM frame                   │
                          │                                              │
                          │  [status_loop] ── 1Hz ──→ status事件         │
                          │    └─ _maybe_rollover()                      │
                          └──────┬───────────────────────────────────────┘
                                 │
                          WebSocket binary 0x11 → 浏览器播放
```