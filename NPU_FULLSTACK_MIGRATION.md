# MOSS-VL Realtime Demo — NPU 全栈适配总结（最终版 vs 原始代码）

> 基线定义（"原始"）：
> ① Demo 仓库 = git HEAD（上游原始代码）
> ② VLM checkpoint = 上游原版 MOSS-VL-Realtime-0708（未打任何 NPU 补丁）
> ③ TTS checkpoint = tianjingcheng tar 包的最终适配版（gpt2_decoder.py.bak_torchair /
>    *_20260724.py / modeling_moss_tts_nano.py 2026-07-24 快照）
> ④ 环境 = 无（新建）；配置 = .env.deploy.example 模板
>
> 中间适配过程的试错（dynamic=True 编译、错误的 bool-bias 数学路径、误入
> build_inference_input_ids 的编辑等）均已回退，不在本清单内。

---

## 一、特性视角

### F1. NPU 设备抽象层（新增，地基特性）
新增 `server/device_compat.py`：设备类型检测（npu/cuda/cpu）、设备字符串生成、
内存监控、attention 后端选择、`ASCEND_RT_VISIBLE_DEVICES` 可见性控制。
全仓所有 `torch.cuda.*` / `"cuda:N"` 硬编码经由此模块间接化。

### F2. VLM 在 NPU 上推理（transformers 兼容 + 算子适配）
- transformers 4.57↔5.x 双版本兼容（checkpoint 内 try/except 与 hasattr 兜底）
- RoPE inv_freq 加载破坏的自修复（4.57 无需、5.x 需要，双保险保留）
- NPU 8 维张量上限：10D permute+reshape 挪到 CPU 执行
- attention 固定 eager（NPU 无 flash-attn）
- `VLM_DEPLOY=inproc`：单卡网关进程内加载，绕开多进程设备可见性

### F3. ASR 迁移 NPU（配置级迁移）
funasr 1.3.1 原生接受 `device="npu:0"`，SenseVoice+fsmn-vad 与 VLM 共享卡 0
（仅 ~1GB HBM）。解码 5-10s(CPU) → 0.1s(NPU)，根治了终稿排队延迟。

### F4. TTS 在 NPU 上推理（torchair 图编译，最大工作量特性）
六个子特性协同（详见文件视角 C 区）：
- a) 图编译钩子：`_decode_local` 静态图（torchair 内置于 torch-npu 2.11）
- b) 形状契约：checkpoint 的 pad-to-bucket 参数透传 + env 可配（320/640）
- c) 流式多音色 warmup：启动时在健康门内完成图编译 + prompt-codes 缓存
- d) prompt-codes 按音色缓存：首包 127ms → 50ms
- e) 长度感知帧数上限：采样失控（无 EOS 撞 320 帧上限）封顶在 4帧/字+32
- f) result 结束事件恢复：修复"音频正常但 job 判失败→模型状态丢弃"级联

### F5. 前端两处缺陷修复
- sampler 竞态：静态图片帧在 session.created 早于 sampler 创建时被静默丢弃
  （"图片没送达→模型按规则沉默"的根因）
- PCM 播放器采样率 48kHz 对齐 TTS 输出（消除 cicici 噪音）

### F6. TTS 背压参数适配快引擎
CPU 引擎 RTF 0.73 的水位（2.5s/1.0s/30s）在 NPU RTF 0.35 下误伤：
高水位秒触发→段持流→段间静音 gap→段超龄丢弃（丢内容）。放宽至 15s/3s/90s。
打断安全不受影响（barge-in 走显式清队列路径）。

### F7. MiniMax 云端 TTS lane（会话级可选引擎）
适配器 + boot 探测 + 会话级切换 + 懒回退本地池。303 音色，与本地 NPU TTS
构成"低延迟本地 / 高音质云端"双 lane。

### F8. 双栈并行架构（A/B 对照与回滚保险）
栈 A 统一环境（:8000/:20941）与栈 B 分开环境（:8001/:20942）并行运行，
`split_stack.sh` 管理；显式 env 转发防 tmux server 陈旧环境快照污染。

---

## 二、修改文件视角

### A. VLM Checkpoint（public/workspace/models/MOSS-VL-Realtime-0708/）
| 文件 | 行数 | 内容 |
|---|---|---|
| modeling_moss_vl.py | ~97 | OutputRecorder 导入 fallback；RoPE "default" 手算 inv_freq；create_causal_mask 4.57/5.x 双签名；_get_initial_cache_position hasattr 兜底；empty_cache NPU 感知；_supports_flash_attn=False |
| configuration_moss_vl.py | 4 | pad_token_id 参数与赋值（5.x __getattribute__ 强检） |
| processing_moss_vl.py | 44 | interpolation 默认值/__init__/Kwarg 字段；10D permute→CPU |
| video_processing_moss_vl.py | 7 | 10D permute→CPU |
| preprocessor_config.json | 1 | + "interpolation": "BICUBIC" |

### B. TTS Checkpoint（extend/MOSS-TTS-Nano/，tianjingcheng 基线之上）
| 文件 | 行数 | 内容 |
|---|---|---|
| MOSS-TTS-Nano/gpt2_decoder.py | +18 | RoPE repeat_interleave→stack+flatten（torchair 无该 GE converter）；4 处 BaseModelOutputWithPast 惰性导入（kernel 命名空间）；RoPE 类 +_base/_dim/_compute_inv_freq 元数据；SDPA 两路径→委托 _eager_attention（npu_fusion_attention_v3 无 GE converter） |
| MOSS-TTS-Nano/modeling_moss_tts_nano.py | ~90 | generate_stream/inference_stream 透传 max_prompt_len/max_total_len（env 桶）；voice_clone 判定接受 prompt_audio_codes；**result 结束事件恢复**（上游删除导致 job-failed 级联） |
| MOSS-Audio-Tokenizer-Nano/modeling_moss_audio_tokenizer.py | 43 | 2 处 SDPA→数学路径 **masked_fill**（bool 掩码语义；此前加法路径=噪音根因） |

### C. Demo 仓库 — 本次 TTS/环境统一会话（vs git HEAD）
| 文件 | 行数 | 内容 |
|---|---|---|
| server/adapters/tts/.../moss_tts_nano_runtime.py | +155 | F4-a 图编译钩子（env 门控/独立缓存目录/kernel namespace 补丁）；F4-d prompt-codes 缓存；F4-e 长度感知帧上限 |
| server/adapters/tts/.../moss_tts_nano_sidecar.py | +62 | F4-c 多音色流式 warmup（MOSS_TTS_NANO_WARMUP_VOICES） |
| src/hooks/useSession.ts | 14 | F5 sampler 竞态修复（sampler 创建提前到首个 await 之前） |
| scripts/deploy/split_stack.sh | 新文件 | F8 双栈管理（start/stop/status + 显式 env 转发） |

### D. Demo 仓库 — 前序 NPU 适配会话（vs git HEAD，一并计入与原始对比）
| 文件 | 行数 | 内容 |
|---|---|---|
| server/device_compat.py | 新文件 | F1 设备抽象层 |
| server/adapters/vlm/moss_vl_hf/adapter.py | 155 | NPU 设备串/attn 选择/inv_freq 修复钩子/加载路径 |
| server/gpu/topology.py | 102 | npu-smi 拓扑探测 + NPU torch child probe |
| server/gpu/placement.py / supervisor.py | 5+5 | npu device_str；ASCEND_RT_VISIBLE_DEVICES |
| server/sidecars.py | 16 | TTS_PYTHON 独立解释器通道 + NPU 可见性注入 |
| server/vlm_worker/app.py | 32 | torch.cuda.* → device_compat.* |
| server/realtime/mossvl_patches.py | 5 | NPU 感知 |
| server/session/orchestrator.py | 16 | device_compat 内存监控 + TTS 链路日志 |
| server/routers/session_ws.py | 4 | 帧到达日志 |
| server/adapters/asr/funasr_sensevoice/adapter.py | 2 | auto 默认 cpu→(配置 npu:0) |
| scripts/deploy/demo.sh | 11 | npu-smi 分支 |
| scripts/deploy/run_backend.sh | ~6 | PYBIN 解释器通道 |
| src/lib/pcmPlayer.ts | 6 | F5 48kHz AudioContext |
| vite.config.ts | ~8 | VITE_BACKEND_ORIGIN 可配代理目标（双栈支撑） |

### E. 配置 .env.deploy（vs 模板）
| 块 | 关键项 |
|---|---|
| 设备 | ASCEND_RT_VISIBLE_DEVICES=0；ASR_DEVICE=npu:0；VLM: ATTN_IMPL=eager + VLM_DEPLOY=inproc + OFFLINE_PROVIDER=none |
| TTS-NPU | TTS_PYTHON=moss_tts_npu env；TTS_SIDECAR_COUNT=1；TTS_GPU=1；MOSS_TTS_NANO_DEVICE=npu:0 + ATTN=sdpa + TORCHAIR=1 + 桶 320/640 + WARMUP(3音色,流式) + checkpoint 指向 extend/ 适配副本 |
| 背压 | AUDIO_BUFFER_HIGH_S=15 / LOW_S=3 / TTS_UNIT_MAX_AGE_S=90 |
| 帧上限 | MOSS_TTS_NANO_LENGTH_AWARE_FRAMES=1 / FRAMES_PER_CHAR=4.0 / MIN_FRAMES=96 |
| MiniMax | API_KEY / BASE_URL(.com) / speech-02-hd / 24000Hz |
| 路径 | 全部指向 sj-ssd3 挂载的实际模型布局 |

### F. 新环境 /opt/mamba/envs/moss_tts_npu（统一栈唯一运行时）
python 3.11.15 · torch 2.11.0+cpu · torch-npu 2.11.0（内置 torchair）·
transformers 4.57.6（demo 官方 requirements 线）· torchvision 0.26.0+cpu ·
torchaudio 2.11.0 · torchcodec 0.11.1 · funasr 1.3.1 · pynini 2.1.7 /
WeTextProcessing 1.2.0 · onnxruntime 1.23.2 · protobuf 5.29.5 ·
sentencepiece · fastapi/uvicorn/python-multipart · accelerate/aiohttp/pillow
（全部公网 pypi 可复现）

### G. 运维产物
- demo_url.txt：双栈访问地址 + 管理命令
- NPU_ADAPTATION_SUMMARY.txt：追加第十一节（sampler 竞态）
- 本文档

---

## 三、最终架构（统一栈）

```
环境 moss_tts_npu (py3.11 / torch-npu 2.11 / transformers 4.57.6)
├── 网关 :8000  ── VLM MOSS-VL-Realtime 22GB  → NPU 卡0 (eager attn, inproc)
│                ├ ASR SenseVoice+fsmn-vad   → NPU 卡0 (共卡, ~1GB)
│                ├ MiniMax speech-02-hd      → 云端 API (可选 lane)
│                └ 编排/WS/持久化             → CPU (纯软件层)
└── TTS sidecar :18100 (TTS_PYTHON 同环境 spawn)
    └── MOSS-TTS-Nano + Audio-Tokenizer → NPU 卡1 (torchair 静态图)
        首包 ~50ms · RTF 0.17-0.35 · warmup 90s(3音量)

web :20941 (vite preview, /api 反代) — tmux 会话 moss (demo.sh)
```

关键指标对照：TTS 首包 CPU 370ms→NPU 50ms；ASR 解码 5-10s→0.1s；
TTS 吞吐 RTF 0.73→0.35；全模型（除 MiniMax 云端）本地 NPU。
