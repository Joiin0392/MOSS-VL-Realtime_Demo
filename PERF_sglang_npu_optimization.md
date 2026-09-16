# sglang NPU 推理性能优化

> 日期：2026-09-06
> 环境：Ascend 910B2C / CANN 9.0.0 / torch_npu 2.10.0.post4 / sglang fork (npu/minimal)
> 模型：MOSS-VL-Instruct-0708

---

## 一、优化前基线

| 指标 | 值 |
|---|---|
| decode 吞吐 | 28.4 tok/s |
| prefill (11 tok) + 1 decode | 0.099s |
| NPU graph | 禁用 (`--disable-cuda-graph`) |
| server warmup | 禁用 (`--skip-server-warmup`) |
| max_running_requests | 2048 (默认，过大) |

---

## 二、优化措施

### 2.1 启用 NPU graph（主要优化）

**问题**：`--disable-cuda-graph` 禁用了 NPU graph，每步 decode 都有完整的 kernel launch 开销。

**修复 1**：移除 `--disable-cuda-graph`，添加 `--cuda-graph-bs 1 2 4 8 16 32`

**修复 2**：`cuda_graph_runner.py` 的 `can_run()` 方法中，`is_encoder_lens_supported` 检查 `torch.all(encoder_lens > 0)`——对纯文本请求（encoder_lens=0）返回 False，导致 graph 无法用于纯文本 decode。修改为 `>= 0`。

```python
# Before:
is_encoder_lens_supported = (
    torch.all(forward_batch.encoder_lens > 0)
    if self.is_encoder_decoder
    else True
)

# After:
is_encoder_lens_supported = (
    torch.all(forward_batch.encoder_lens >= 0)
    if self.is_encoder_decoder
    else True
)
```

**理由**：纯文本 decode 时 cross-attention 被跳过（`skip_cross_attention=True`），encoder_lens=0 不会影响 graph replay 的正确性。graph 在 capture 时已包含 cross-attention 路径（`get_is_capture_mode() → skip=False`），replay 时 encoder_lens=0 → cross-attn 输出被 `full_text_row_masked_out_mask` 置零 → 等价于跳过。

### 2.2 启用 server warmup

移除 `--skip-server-warmup`，让 sglang 启动时自动 warmup（首次请求不再冷启动）。

### 2.3 限制 max_running_requests

添加 `--max-running-requests 32`，从默认 2048 降到 32。单卡场景 2048 过大，浪费内存池资源。

---

## 三、优化后结果

| 指标 | 优化前 | 优化后 | 提升 |
|---|---|---|---|
| decode 吞吐 | 28.4 tok/s | **46.3 tok/s** | **+63%** |
| NPU graph | 禁用 | 启用 (bs 1-32) | — |
| 输出质量 | 正常 | 正常 (prompt_tokens=11) | 无退化 |

---

## 四、改动清单

| 文件 | 改动 |
|---|---|
| `cuda_graph_runner.py` | `can_run()`: `encoder_lens > 0` → `>= 0` |
| `.env.deploy` | 移除 `--disable-cuda-graph`、`--skip-server-warmup`；添加 `--cuda-graph-bs 1 2 4 8 16 32`、`--max-running-requests 32` |

---

## 五、未采用的优化（已评估）

| 优化 | 原因 |
|---|---|
| piecewise CUDA graph | MossVLForConditionalGeneration 在 `is_piecewise_cuda_graph_disabled_model` 列表中，框架层硬禁用 |
| torch.compile | 需要 torchair/torch_npu.compile 支持，复杂度高，稳定性未知 |
| 算子级优化 | 用户要求暂不考虑 |
