# PR #15 Reviewer 反馈分析：cross_attention_mask 跳过是否导致训推不一致

> PR: https://github.com/OpenMOSS/MOSS-VL/pull/15
> Reviewer: SSSSuperC
> 反馈原文："直接这样跳过cross_attention_mask 在prefill阶段 还是会有训推不一致的问题吧 虽然也是能正常输出 不用flash_infer的话 完全没办法用custom_mask吗 如果可以的话 最好还是在prefill阶段用上cross_attention_mask吧"

---

## 一、为什么做这个修改

### 背景

MOSS-VL 的 sglang fork 在 `MossVLForConditionalGeneration.prepare_forward_batch()` 中构建 frame-level causal mask（`cross_attention_custom_mask`），用于控制 cross-attention 中每个 text token 能看到哪些 video frame。

这个 mask 是 **FlashInfer 专有格式**（packed 1D uint8），只有 FlashInfer backend 能消费。Ascend NPU 没有 FlashInfer，使用的是 CANN ascend backend。

### 修改内容

在 `prepare_forward_batch()` 中增加 guard：当 prefill backend 不是 FlashInfer 时，跳过 mask 构建：

```python
server_args = get_global_server_args()
prefill_backend, _ = server_args.get_attention_backends()
if prefill_backend != "flashinfer":
    return  # 跳过 custom mask
```

### 原因

ascend backend 不支持 FlashInfer 的 packed custom mask 格式。如果不跳过，构建的 mask 会被设置到 `forward_batch.cross_attention_custom_mask` 上，但 ascend backend 无法消费它——要么忽略（无害但浪费计算），要么报错。

---

## 二、mask 的作用机制

### 训练时（HF）

HF processor 在 `processing_moss_vl.py` 的 `_create_cross_attention_mask()` 中生成 mask：

- shape: `(B, 1, text_len, num_frames)`
- 语义: True = masked（不可见），False = visible（可见）
- 逻辑: `visible_mask = cum_image_tokens.unsqueeze(-1) > frame_indices`
  - text token 在位置 t 的 cum_image_tokens = 该位置之前出现了多少个 image token
  - frame i 在 cum_image_tokens > i 时可见（即 text token 在第 i+1 个 image token 之后）

mask 在 HF forward 中通过 `cross_attention_mask` 参数传入 cross-attention 层，作为 attention mask 使用。

### 推理时（sglang GPU / FlashInfer）

sglang fork 将 mask 转换为 FlashInfer 的 packed 1D 格式（`cross_attention_custom_mask`），FlashInfer backend 在 cross-attention 计算时应用它——行为与训练一致。

### 推理时（sglang NPU / ascend，mask 被跳过）

cross-attention 使用标准 non-causal 模式（`causal=False`），所有 text token 看到所有 vision tokens。**没有 frame-level causal 限制**。

### 另一个 mask：full_text_row_masked_out_mask

这是 **token-level** mask（不是 frame-level），控制"这个 text token 能不能看到任何 vision token"。如果看不到任何 vision token，cross-attention 的输出被置零。

这个 mask 是逐元素乘法（`hidden_states = mask * hidden_states`），不依赖任何 backend，**在 NPU 上正常工作**。

---

## 三、训推一致性分析

### 单图场景（1 frame）

| text token 位置 | 训练 mask | NPU 推理（无 mask） | 一致？ |
|---|---|---|---|
| image token 之前 | masked（不可见） | full_text_row_masked_out_mask 置零 | ✅ 一致 |
| image token 之后 | visible（可见） | 全部可见 | ✅ 一致 |

单图只有 1 个 frame，cum_image_tokens 在 image token 之后为 1，`1 > 0 = True`（可见），mask 全为 False（不遮蔽）。跳过 mask 后全部可见——**与训练一致**。

### 视频场景（N frames, N > 1）

以 2 帧（frame0 在 pos 4, frame1 在 pos 7）为例：

| text token 位置 | 训练 mask | NPU 推理（无 mask） | 一致？ |
|---|---|---|---|
| pos 0-3（frame0 之前） | 两帧都 masked → 置零 | 全不可见 → 置零 | ✅ 一致 |
| pos 4-6（frame0 后, frame1 前） | frame0 可见, frame1 **masked** | 两帧**都可见** | ❌ **不一致** |
| pos 7-9（frame1 后） | 两帧都可见 | 两帧都可见 | ✅ 一致 |

**不一致出现在 frame0 和 frame1 之间的 text token**：训练时这些 token 看不到 frame1（未来帧），但 NPU 推理时能看到。

### 纯文本场景（0 frame）

无 vision input，cross-attention 层被跳过（`skip_cross_attention=True`），mask 完全不涉及。**一致**。

---

## 四、结论

| 场景 | 训推一致？ | 原因 |
|---|---|---|
| 纯文本 | ✅ | 无 vision input，cross-attention 跳过 |
| 单图 | ✅ | 1 frame，mask 退化为全可见 |
| 多帧视频 | ❌ | text token 能看到未来帧（训练时被 mask 遮蔽） |

**SSSSuperC 的担心是正确的**——对于多帧视频输入，跳过 cross_attention_mask 会导致训推不一致。

**但影响范围有限**：
- 不影响纯文本推理（已通过 `_build_mm_items` 修复解决了 pad 污染问题）
- 不影响单图推理（最常见的多模态场景）
- 仅影响多帧视频推理中"帧间 text token"的 cross-attention（这些 token 在训练时看不到未来帧，推理时能看到）

---

## 五、解决方案

### 方案 A：在 ascend backend 上实现 custom mask（理想但复杂）

ascend backend 的 cross-attention 走 `run_sdpa_forward_extend`（torch native sdpa），可以传入 `attn_mask` 参数。将 packed 1D mask 转换为 sdpa 兼容的 2D/3D mask 格式即可。

**实现思路**：
1. `prepare_forward_batch` 不跳过，构建 mask（即使 backend 不是 FlashInfer）
2. 在 ascend backend 的 `run_sdpa_forward_extend` 中，当 `is_cross_attention=True` 且 `cross_attention_custom_mask` 非 None 时，将 packed 1D mask reshape 为 `(q_len, kv_len)` 的 2D mask，传给 `F.scaled_dot_product_attention` 的 `attn_mask` 参数

**优点**：完全一致的训推行为
**缺点**：需要修改 ascend backend 的 cross-attention 路径

### 方案 B：在 sglang model 层用 torch sdpa 替代 ascend backend 的 cross-attention（折中）

在 `MossVLTextCrossAttention.forward()` 中，当检测到是 cross-attention 且有 custom_mask 时，不走 RadixAttention（ascend backend），直接用 `F.scaled_dot_product_attention` + mask 计算。

**优点**：不改 ascend backend，只在 model 层处理
**缺点**：cross-attention 不走 paged KV cache，需要单独收集 K/V

### 方案 C：保持现状，在文档中标注限制（最小改动）

在 PR 描述中明确说明：NPU 推理的多帧视频场景存在 frame-level mask 不一致，单图和纯文本不受影响。

**优点**：零代码改动
**缺点**：视频推理质量可能下降

---

## 六、验证

### 已验证（通过代码分析和数值模拟）

1. **单图 mask 退化为全可见**：cum_image_tokens=1, frame_indices=[0], `1 > 0 = True`（visible）→ mask 全 False → 跳过 mask 无影响 ✅
2. **视频 frame 间不一致**：pos 4-6 的 `cum_image_tokens=1`, `frame_indices=[0,1]`, `1 > 1 = False`（frame1 masked）→ 跳过 mask 后 frame1 变为可见 ❌
3. **full_text_row_masked_out_mask 在 NPU 上正常工作**：逐元素乘法，不依赖 backend ✅

### 待验证（需要多帧视频输入）

- 多帧视频在 NPU 上的实际输出质量对比（有 mask vs 无 mask）
- 需要准备一个多帧视频测试用例

---

## 七、建议

**推荐方案 A**（在 ascend backend 实现 custom mask），因为：
1. SSSSuperC 的反馈明确要求"最好还是在 prefill 阶段用上 cross_attention_mask"
2. 实现不复杂：packed 1D mask → reshape 为 2D → 传给 sdpa 的 attn_mask
3. 彻底解决训推一致性问题
4. 不影响纯文本和单图（mask 在这两种情况下退化为全可见，传与不传结果相同）

**实施时机**：可作为 PR #15 的后续 commit 补充，或单独提 PR。
