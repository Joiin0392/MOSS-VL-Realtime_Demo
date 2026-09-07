# VLM 显存与并发

## 测量范围

以下只统计 VLM 后端，不包含 memory 摘要模型、ASR 或 TTS。
测试模型为 [MOSS-VL-Realtime-SGLANG](https://huggingface.co/OpenMOSS-Team/MOSS-VL-Realtime-SGLANG)，
BF16、单卡、CUDA Graph 开启、async decode 关闭，后端代码 `22b671a`。
每路 1 FPS、最长边 512、JPEG quality 60、生成目标 4 tokens/s；
视觉 raw window 60 秒，pooling 关闭，context 131,072。
Demo 开启文本/图像 memory，CPU 编码，idle 8K / hard 12K rollover。

测试使用 H200。缩减预算实验把 PyTorch 分配上限设为 78 GiB，
并按 `78 GiB * 0.60` 计算静态预算，得到约 25.4 GiB 的 KV 池；
同时记录整个 VLM 进程的 NVML 显存，以覆盖 PyTorch 以外的占用。
这验证 80GB 级别的显存容量，不代表已经测过真实 A100/H100 的吞吐。
本文统一用 GiB（2^30 字节）统计，避免与十进制 GB 混用。

## 不要用进程显存除以会话数

一个后端实例内，模型权重共享，KV 池预先分配，会话从池内领取 slots。
因此关闭 session 后，KV slots 会回到空闲池，但 `nvidia-smi` 占用通常不会下降。

| 项目 | 实测含义 |
| --- | --- |
| 共享权重及加载时常驻张量 | 约 21.3 GiB，每个后端实例一份，不是每路一份 |
| 原 H200、static fraction 0.60 的 KV 池 | 61.7 GiB，336,874 slots |
| 78 GiB 预算、static fraction 0.60 的 KV 池 | 25.4 GiB，138,662 slots |
| 原配置 4 路、20 分钟的实际 KV 峰值 | 合计 13.1 GiB，不是 61.7 GiB |
| 原配置长测单路 KV 分布 | 中位约 2.53 GiB，P95 约 3.36 GiB，最大约 3.69 GiB |
| 缩减预算单路、180 秒短测 | KV 峰值约 2.07 GiB，完成一次 rollover，关闭后池完全回收 |
| 缩减预算 4 路、20 分钟长测 | KV 峰值约 12.97 GiB，VLM 进程 NVML 峰值约 50.33 GiB |

单路分布取输入开始 60 秒后的观测，包含 rollover 后重新填充窗口的阶段；
不同视频形状、文本长度及所处阶段会改变这个数，不能把均值当作硬上限。

当前池的实际 K/V buffers 是 48 层、8 个 KV heads、head dimension 128、BF16：

```text
每 slot = 48 * 8 * 128 * 2 (K+V) * 2 bytes = 192 KiB
每路实际 KV = (仍保留的视觉 slots + 文本 decoder slots) * 192 KiB
```

CUDA Graph、图像编码临时张量、分配器缓存和 CUDA runtime 是另外的开销。
进程占用与实际 session KV 不是同一个指标。

这里按实际分配的 48 层统一池计算，而不是理想的紧凑布局。模型包含
12 层 cross-attention 和 36 层 self-attention，视觉与文本 slots 仍从
同一个池分配；拆成分别匹配层数的池可能进一步节省空间，但涉及引擎
和映射契约的改造，本次没有修改这一布局。

## 配置判断

原 H200 的 0.60 配置没有显示 KV 泄漏，但如果只服务当前 4 路，61.7 GiB
的池明显留有余量：长测峰值仅使用约 21%。更大的池能容纳更多或更重的
请求，但不会让固定的 4 路自动变快。

不能把池任意缩到 13 GiB：当前启动检查要求至少能容纳一个完整 128K
context，仅这一项需要 24 GiB KV。缩减预算的 25.4 GiB 池已经接近这个
启动下限。H200 上约 0.34-0.35 的静态比例与本次缩减预算接近；
实际设置应以启动日志中的 pool slots、权重占用和可用显存为准。

视觉滑窗回收物理 KV，却不清除历史 context 位置；文本 KV 也会增长，
因此这个容量结论依赖 memory rollover 正常运行。4 路都完整保留 128K
物理 KV 则仅 KV 就需要 96 GiB，不能用本测试证明这种负载也能放进 80GB。
分辨率、FPS、视觉窗口长度改变时应重新测量。

## 80GB 级别容量结果

缩减到 78 GiB 预算、静态比例 0.60 后，4 路连续运行 20 分钟：

- 每路完成两次自然 rollover，4 路均存活并持续处理视频。
- 4,800 个摄像头输入帧中转发 4,790 个（99.79%）；切换期间不是零丢帧。
- 模型收到的 4,962 帧全部处理完成，包含问题附带帧和 rollover 恢复帧。
- memory 写入丢弃为 0，结束时队列为 0；关闭后 138,662 个 KV slots 全部归还。
- 单路 KV 中位约 2.55 GiB、P95 约 3.36 GiB、最大约 3.69 GiB。
- VLM 进程 NVML 峰值 50.33 GiB；PyTorch reserved 峰值 49.53 GiB、
  allocated 峰值 48.84 GiB。这些口径不能相加，实际 session KV 已包含在池内。

因此，只计算 VLM、保持本页输入和 memory 配置时，80GB 级别的显存可以
支持当前 4 路需求。同样的 `--mem-fraction-static 0.60` 会在较小的卡上
重算池大小，不会沿用 H200 的 61.7 GiB 池。部署时显式设置 context 131,072
及后端 `--max-running-requests 4`、Demo `SGLANG_OMNI_SESSIONS_PER_REPLICA=4`，
并确认启动日志中的 pool slots 不少于 131,072。具体 80GB 卡的吞吐和
软件兼容性仍需在实机验证；测试没有改动线上配置或重启线上服务。

质量探针单独计量：4 路 memory 均保留自己的标记，未检测到跨会话标记。
最后一次标记问答中 3 路准确回显，另一路在 10 秒观察窗口内未回显，
因此本次容量/生命周期通过不等于语义质量 4/4 通过，未回显的原因尚未确定。

## 连接池修复

旧实现用同一个 `used` 同时表达本地握手预留和推测的远端满额占用。
探活线程可能将握手预留误当成旧满额标记清零，导致实际已连接，网关
却报告空闲 slot。

现在分别记录已建立的 `sessions`、握手中的 `pending`、以及
`remote_full_until` 冷却期限。探活和容量重试不能减少本地预留；
成功握手将预留转换为已建立会话，失败只释放自己的预留，重复 stop
不会多次释放。状态接口增加 `pending`、`tracked_sessions` 便于观测。

原 bug 的真实探活线程用例在旧版本失败，修复后通过。仓库统一入口
`python scripts/run_tests.py` 的 29 个测试套件通过；包含新增用例的
连接池/适配器定向回归共 28 项通过。
