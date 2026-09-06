# MOSS-VL Realtime Demo

MOSS-VL 实时视频/语音交互 Demo，以及供外部客户端接入的薄网关。浏览器界面使用 React/Vite，服务端使用 FastAPI；实时 VLM 可以运行在独立的 SGLang-Omni 后端，也保留原有 HF worker 和远端已合入的 NPU 路径。

本仓库负责交互、会话编排和协议适配，**不是模型权重仓库，也不是 SGLang-Omni 推理引擎本身**。

## 仓库与版本

| 组件 | 链接 | 用途 |
| --- | --- | --- |
| Demo / 薄网关 | [fnlp-vision/MOSS-VL-Realtime_Demo](https://github.com/fnlp-vision/MOSS-VL-Realtime_Demo) | 当前仓库，前端、ASR/TTS 编排、adapter、REST/WS 网关 |
| 实时推理后端 | [fnlp-vision/sglang-omni-realtime](https://github.com/fnlp-vision/sglang-omni-realtime) | MOSS-VL 特化的持续请求调度、视觉 KV、TP、流式推理 |
| 后端官方上游 | [sgl-project/sglang-omni](https://github.com/sgl-project/sglang-omni) | 特化后端基于该项目开发 |
| 后端旧 fork | [CloudRipple/sglang-omni](https://github.com/CloudRipple/sglang-omni) | 历史来源，不是本 Demo 的推荐安装入口 |
| TF 5.12.1 兼容模型 | [OpenMOSS-Team/MOSS-VL-Realtime-SGLANG](https://huggingface.co/OpenMOSS-Team/MOSS-VL-Realtime-SGLANG) | 与特化后端配套的 checkpoint、processor 和自定义代码；当前为私有仓库，需授权 |
| 原版模型 | [OpenMOSS-Team/MOSS-VL-Realtime](https://huggingface.co/OpenMOSS-Team/MOSS-VL-Realtime) | 原始权重及 Transformers 4.57 系列参考实现，供 HF 路径使用 |

2026-09-06 的配套版本为：后端 [`115b1e2`](https://github.com/fnlp-vision/sglang-omni-realtime/commit/115b1e2b7c477187a0138fe2fbac953368779a07)，TF 5.12.1 模型包 [`bcfd9cc`](https://huggingface.co/OpenMOSS-Team/MOSS-VL-Realtime-SGLANG/tree/bcfd9ccf1e9db2896ad852301cc8dde4a6349c78)，原版模型的 Query RoPE 修复 [`1e6a45b`](https://huggingface.co/OpenMOSS-Team/MOSS-VL-Realtime/commit/1e6a45b292eeaf02aa733bd3aa7b6c85214ddc86)。正式部署应固定实际使用的代码和模型 revision，而不是只记录模型名称。

## 本轮更新

- 新增 `VLM_DEPLOY=sglang_omni`：连接独立实时推理实例，支持多副本、每副本多 slot、探活和有限故障切换。
- 修复软打断、旧输出隔离、输入错误/ACK 超时恢复，接入可协商的 `session.usage`，按实际上下文余量触发有配置前提的 Demo rollover。
- 补齐薄网关：创建/取消清理、reset 代次隔离、旧凭证撤销、有效帧上限、心跳、健康探测、模型版本观测和对账日志。
- 保留远端 NPU 基础设施和 HF adapter 改动；启动脚本同时支持自定义 `PYBIN` 与 WS 帧限制/心跳参数。
- 增加可直接修改的 [SGLang-Omni 最小配置](./.env.deploy.sglang-omni.example)，显式声明 `requests` 依赖，并将测试依赖独立放入 [requirements-dev.txt](./requirements-dev.txt)。

## 三个协议平面

```text
浏览器 Demo
  -> /api/sessions + /api/session/{id}/ws
  -> 会话编排 / 可选 ASR、TTS、memory
  -> SGLang-Omni adapter
  -> 独立后端 /v1/video/realtime

外部协议客户端
  -> REST /v1/realtime/sessions + WS /v1/realtime?ws_token=...
  -> 薄网关 / 粘性路由 / 一次性凭证
  -> 独立后端 /v1/video/realtime
```

| 平面 | 职责与边界 |
| --- | --- |
| Demo `/api/...` | 浏览器协议、语音和字幕、可选记忆与恢复；与原始模型协议不同 |
| 薄网关 `/v1/realtime...` | 转发模型协议，提供 REST 生命周期和短期凭证；不提供 Demo 的 memory/grace/replay |
| 后端 `/v1/video/realtime` | 模型会话与增量推理；不负责浏览器、客户鉴权或计费 |

Demo 和薄网关当前分别维护池占用计数。若两者同时共享相同后端副本，其本地 slot 不是统一全局配额，后端仍会拒绝超额会话；正式容量规划应分配独立实例池或在平台统一准入。不要将两边显示的容量相加。

## 环境隔离

**不要把 Demo、实时后端和 TTS 引擎的依赖装进同一个虚拟环境。**

| 环境 | 依赖来源 | 说明 |
| --- | --- | --- |
| Demo `.venv` | [requirements.txt](./requirements.txt) | Python 3.12；Torch 2.8 / Transformers 4.57.1，用于主服务、原 HF 路径及语音适配 |
| 实时后端独立 `.venv` | [后端 pyproject.toml](https://github.com/fnlp-vision/sglang-omni-realtime/blob/main/pyproject.toml) | SGLang 0.5.16 / Transformers 5.12.1 / Torch 2.11；通过 HTTP/WS 与 Demo 通信 |
| 离线聊天 `.venv-sglang` | [requirements-sglang.txt](./requirements-sglang.txt) | 历史离线聊天引擎，不是 realtime 后端 |
| 可选 TTS 环境 | [requirements-vllm.txt](./requirements-vllm.txt)、[requirements-mossrt.txt](./requirements-mossrt.txt)、[requirements-cosyvoice.txt](./requirements-cosyvoice.txt) | 按所选 provider 单独安装 |

CUDA quickstart 面向 Linux/NVIDIA，需兼容驱动、FFmpeg/torchcodec 所需动态库和 Node 20.19+ 或 22.12+。NPU 保留已有代码路径，但需平台匹配的 Torch/torch_npu 环境及 `PYBIN`，不要在 NPU 环境照装 CUDA wheel；本次发布不代表重新完成 NPU 端到端验收。

## Quickstart：SGLang-Omni + 视频/文本 Demo

先建立最小视频/文本链路，再按需开启语音和 memory。以下端口均为示例，应避免与现有服务冲突。多机部署时，把 loopback 地址替换为内部可达地址，并配置访问控制。

### 1. 启动独立推理后端

在单独的终端/环境中执行：

```bash
git clone https://github.com/fnlp-vision/sglang-omni-realtime.git
cd sglang-omni-realtime
git switch --detach 115b1e2b7c477187a0138fe2fbac953368779a07
uv venv .venv -p 3.12
source .venv/bin/activate
uv pip install -e .

hf download OpenMOSS-Team/MOSS-VL-Realtime-SGLANG \
  --revision bcfd9ccf1e9db2896ad852301cc8dde4a6349c78 \
  --local-dir /path/to/moss-vl-realtime-sglang

python examples/run_moss_vl_realtime_server.py \
  --model-path /path/to/moss-vl-realtime-sglang \
  --gpu 0 --host 127.0.0.1 --port 18500 \
  --context-length 131072 --mem-fraction-static 0.60 \
  --max-running-requests 1
```

需预先安装 `uv`，并通过自己的 Hugging Face 授权访问私有模型；不要把 token 写入仓库或配置示例。后端依赖包含 CUDA 13 组件，环境前提见[后端 README](https://github.com/fnlp-vision/sglang-omni-realtime#readme)。128K/0.60 是起步配置示例，不保证适合所有设备；启动预热结束且 `/health` 正常后再启动 Demo。

```bash
curl --fail http://127.0.0.1:18500/health
curl --fail http://127.0.0.1:18500/v1/models
```

TP 使用 `--tp-size 2 --gpus 0,1` 替代 `--gpu 0`。多个副本应分别监听不同端口/地址，每个副本使用已规划的 GPU。

### 2. 安装 Demo

在另一个终端中执行，避免继承后端虚拟环境：

```bash
git clone https://github.com/fnlp-vision/MOSS-VL-Realtime_Demo.git
cd MOSS-VL-Realtime_Demo
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
npm ci
npm run build
```

本例的 CUDA wheel 只针对 Demo 主环境，不改变后端环境。远程 realtime 模式不在 Demo 进程加载 VLM，也不要求为了该模式额外编译 HF 的 FlashAttention；保留完整主环境依赖以支持现有媒体/语音模块。原有 `scripts/build_venv*.sh` 包含历史内部镜像或 wheelhouse 默认值，外部机器需先检查其配置，不能直接当作通用安装脚本。

### 3. 配置联动

新安装可复制最小配置；已有 `.env.deploy` 时请合并所需字段，不要覆盖原配置：

```bash
cp .env.deploy.sglang-omni.example .env.deploy
```

编辑 `.env.deploy`，至少核对：

```dotenv
VLM_DEPLOY=sglang_omni
SGLANG_OMNI_URLS=http://127.0.0.1:18500
SGLANG_OMNI_SESSIONS_PER_REPLICA=1
SGLANG_OMNI_CONTEXT_LENGTH=131072
MODEL_PATH=/path/to/moss-vl-realtime-sglang
GATEWAY_MODEL_VERSION=model-bcfd9cc_backend-115b1e2
```

示例版本标识对应上面的固定版本；使用不同产物时应填写真实发布标识。`MODEL_PATH` 在该模式主要供 tokenizer/状态估算使用，不会在 Demo 中加载这份 5.12.1 模型。若两层不共享文件系统，应在 Demo 侧提供匹配的 tokenizer 文件。

`.env.deploy` 是字面量 `KEY=VALUE` 配置，**不会展开 `$HOME`、`$REPO` 等变量**，路径请写完整。命令行环境变量优先于该文件；全量参考见 [.env.deploy.example](./.env.deploy.example)，其中 `<repo>` 只是说明用占位符。

| 参数 | 配合要求 |
| --- | --- |
| `SGLANG_OMNI_URLS` | 逗号分隔 HTTP(S) 基础地址，不要附加 `/v1/video/realtime` |
| `SGLANG_OMNI_SESSIONS_PER_REPLICA` | 与各后端 `--max-running-requests` 一致；parked 仍占 slot |
| `SGLANG_OMNI_CONTEXT_LENGTH` | 与后端 context 一致；无 usage 能力时用于回退估计 |
| `SGLANG_OMNI_CONTEXT_RESERVE_TOKENS` | 为在途输入预留空间；默认 4096，不是额外可用 context |
| `GEN_MAX_TOKENS_PER_TURN` | token/秒目标，默认 4，不是每轮总输出长度 |
| `GATEWAY_MAX_FRAME_BYTES` / `WS_MAX_SIZE` | 默认 32 MiB / 64 MiB；传输上限必须大于应用上限 |
| `GATEWAY_MODEL_VERSION` / `GATEWAY_MODEL_VERSIONS` | 版本观测标识，不代表已经实现按客户指定版本路由 |

最小配置关闭 ASR、TTS 和 memory。未配置 memory rollover 时，接近 context 上限会明确结束/报错，不会自动获得无限长会话能力。

### 4. 启动 Demo API 和前端

分别在两个 Demo 终端启动：

```bash
bash scripts/deploy/run_backend.sh
```

```bash
bash scripts/deploy/run_web.sh
```

API 默认监听 `127.0.0.1:8000`，前端 preview 默认端口为 `20941`。浏览器打开 `http://localhost:20941`；摄像头/麦克风在远程机器访问时需要 HTTPS 等安全上下文。

前端只代理 `/api` 到 `VITE_BACKEND_ORIGIN`，**不会自动代理薄网关 `/v1`**。外部协议客户端应访问 API 地址，或由部署反向代理显式配置 `/v1` 的 HTTP/WS 转发。

已有 tmux 的环境也可用 `bash scripts/deploy/demo.sh up`、`status`、`logs api`、`down` 管理本仓库的 API/web。它不会替代上述独立 realtime 后端的启动与版本管理；`start_demo.sh`、`scripts/gpu/*` 是历史集群辅助脚本，带有特定路径/SSH/GPU 假设，不作为通用部署入口。

```bash
curl --fail http://127.0.0.1:8000/api/status
curl --fail http://127.0.0.1:8000/v1/realtime/health
```

检查 Demo 的 `loaded` 状态及池中实际可用副本，不能只凭 HTTP 200 判断模型容量已就绪。

## 开启语音、记忆或原 HF 路径

**ASR。** 下载 [SenseVoiceSmall](https://modelscope.cn/models/iic/SenseVoiceSmall) 和 [FSMN-VAD](https://modelscope.cn/models/iic/speech_fsmn_vad_zh-cn-16k-common-pytorch)，配置 `SENSEVOICE_MODEL`、`SENSEVOICE_VAD_MODEL`、`ASR_DEVICE` 并开启 `ASR_ENABLED=1`。为服务显式规划设备，避免抢占推理 GPU。

**TTS。** 按 provider 选择独立引擎环境及模型，例如 [MOSS-TTS-Nano](https://huggingface.co/OpenMOSS-Team/MOSS-TTS-Nano-100M) 与 [Audio Tokenizer Nano](https://huggingface.co/OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano)，或 [MOSS-TTS-Realtime](https://huggingface.co/OpenMOSS-Team/MOSS-TTS-Realtime) 与 [Audio Tokenizer](https://huggingface.co/OpenMOSS-Team/MOSS-Audio-Tokenizer)。配置 `TTS_PROVIDER`、模型路径和设备，再开启 `TTS_ENABLED`；`TTS_SPAWN=1` 表示由本服务拉起相应 sidecar，连接既有服务时用 `TTS_SPAWN=0` 并填写对应地址。不要将 vLLM/Omni TTS 的依赖合装到 `.venv`。

**Memory。** Demo 的检索与 rollover 需要启用并配置 memory，摘要可来自离线引擎或独立 pi 服务。相关配置及降级策略见 [ops runbook](./docs/ops_runbook.md) 和 [rollover 实现](./server/memory/rollover.py)。它重建的是选取后的上下文，不是无损保留全部历史，也不适用于薄网关。新后端通过 `session.usage` 提供实际预算，旧后端只能回退估计。

**HF worker。** 使用 `VLM_DEPLOY=workers` 和原版 4.57 系列模型；`VLM_WORKER_GPUS` 指定设备。不要让这个环境直接加载 TF5.12 专属自定义代码。NPU 路径保留远端实现，按平台要求安装依赖并用 `PYBIN` 选择解释器；CUDA、NPU、SGLang-Omni 是需要分别验证的部署路径。

## 薄网关接入

配置非空 `SGLANG_OMNI_URLS` 后启用这些端点：

| 端点 | 用途 |
| --- | --- |
| `POST /v1/realtime/sessions` | 创建并预留后端会话，返回 `session_id`、`ws_url`、`ws_token` 和版本信息 |
| `GET /v1/realtime/sessions/{id}` | 查询状态与观测计数 |
| `POST /v1/realtime/sessions/{id}/reset` | 保留网关 ID，重建后端连接，撤销旧 token 并返回新 token |
| `DELETE /v1/realtime/sessions/{id}` | 销毁并释放资源 |
| `WS /v1/realtime?ws_token=...` | 一次性凭证接入，随后使用后端的实时事件协议 |
| `GET /v1/realtime/health`、`/v1/realtime/metrics` | 实例健康、容量和 JSON 指标 |
| `GET /v1/models` | 后端模型元数据 |

内部联调可先创建会话：

```bash
curl --fail -X POST http://127.0.0.1:8000/v1/realtime/sessions
```

客户端将返回的 `ws_token` 加到 `ws_url` 查询参数后建立 WS，等待 `session.created`，发送 `session.configure`，再按 ready/accepted/processed 顺序提交帧。帧与 prompt 共用连续 `seq_no`；`response.done`/`session.done` 是会话级结束，不能当作每段主动回答的边界。

此最小内部网关**不实现 Bearer API Key、客户归属鉴权、客户额度或业务限流**。正式开放必须在平台层补齐，尤其要校验 GET/reset/DELETE 的 session 归属。不要把内部无鉴权示例直接暴露到公网。

默认创建 deadline 30 秒、token TTL 60 秒、等待 attach 90 秒。断连会销毁薄网关会话；没有 Demo 的 grace/replay。容量满与副本不可达分别返回 `session_capacity_exceeded` 和 `no_available_replica`。详细行为见[网关契约](./docs/gateway_contract.md)，运维与计量见[运行手册](./docs/ops_runbook.md)和[告警说明](./docs/gateway_alerting.md)。

## 验证与发布边界

```bash
python -m pip install -r requirements-dev.txt
python scripts/run_tests.py
python scripts/dev/check_env.py --check
npm run build
```

测试入口会为每个文件启动独立进程、清除部署配置并隐藏 GPU；旧脚本式套件按其原有 main 入口执行，其余使用 pytest。不要直接将这些混合入口统一当作 pytest fixtures。日志目录在结束时打印，可用 `--output-dir` 指定。

`server/tests` 包含 adapter、协议、生命周期、配置和假后端测试，不等同于 GPU 模型质量、NPU 或公网平台 E2E 验收。已有真实模型/长会话测试与历史负载报告也有各自输入、版本及硬件条件，不能直接作为任意分辨率、FPS、时长和并发下的 SLA。`reports/` 保留历史开发证据，不是当前上线承诺；高压短会话完成超时仍需按最终部署条件复测。

代码更新后，按“模型与推理后端版本核对 -> 后端重新加载 -> Demo 重启 -> 联调验收”的顺序部署；推送 Git 不会使已有进程自动加载补丁。生产配置 `.env.deploy`、权重、环境和日志不随 Git 同步。

## 代码导航

| 目录 | 内容 |
| --- | --- |
| [server/adapters/vlm/moss_vl_sglang_omni](./server/adapters/vlm/moss_vl_sglang_omni/) | 同步 WS 客户端、会话适配与实例池 |
| [server/gateway](./server/gateway/) | 独立薄网关 REST/WS、token、计量和生命周期 |
| [server/session](./server/session/) | Demo 会话与语音/字幕编排 |
| [server/memory](./server/memory/) | 记忆、摘要与 rollover |
| [server/device_compat.py](./server/device_compat.py) | 远端合入的 CUDA/NPU 兼容层 |
| [src](./src/) | React 前端 |
| [scripts/deploy](./scripts/deploy/) | API/web 启动与环境透传 |
| [server/tests](./server/tests/) | 工程回归 |

MOSS-VL 模型与研究资料见 [OpenMOSS/MOSS-VL](https://github.com/OpenMOSS/MOSS-VL)。推理框架的上游归属和许可证见 [SGLang-Omni](https://github.com/sgl-project/sglang-omni)；模型权重、框架及 vendored 第三方组件分别遵循各自许可证。
