# MOSS-VL-Instruct 离线纯文本推理精度问题：全流程排查与修复

> 日期：2026-09-04 ~ 2026-09-06
> 环境：Ascend 910B2C ×8 / CANN 9.0.0 / torch_npu 2.10.0.post4 / sglang fork (npu/minimal)
> 模型：MOSS-VL-Instruct-0708

---

## 一、问题现象

### 1.1 用户报告

MOSS-VL-Instruct 模型在离线方式（sglang serving）下，纯文本推理精度出现问题：

- **刷新界面后首次请求答复不正确**：输出大量重复内容（如"历史悠久"×40）
- **"介绍一下中国历史"**：前段正确（"中国历史可以分为以下几个时期：\n\n1."），后段出现重复/塌缩（"龟龟龟龟..."）
- **强制结束后再请求正常**：同一实例上，第一次请求出错，中断后第二次请求恢复正常

### 1.2 实测确认

通过直连 sglang `/generate` 端点复现：

```
prompt: <|im_start|>user\n介绍一下中国历史<|im_end|>\n<|im_start|>assistant\n
input_ids: [151644, 872, 198, 109432, 58695, 100022, 151645, 198, 151644, 77091, 198]  (11 tokens)

greedy (max_new_tokens=150):
  输出: "中国历史可以分为以下几个主要时期：\n\n1."
  finish: stop (matched 151645 = <|im_end|>)
  → 第 11 个 token 即 EOS 早停

ignore_eos (max_new_tokens=400):
  输出: "中国历史可以分为以下几个主要时期：\n\n1.龟龟龟龟龟龟龟..."
  → 第 21 步塌缩进单字重复

HF eager 同 prompt 同参数：
  输出: 完整列举夏商周秦汉... 200+ token 正常
```

### 1.3 关键特征

| 特征 | 值 |
|---|---|
| 必现性 | 确定性必现（同 prompt 同 greedy 每次相同结果） |
| 分歧点 | 前 10 步与 HF 逐 token 一致，step 10 分歧 |
| 多模态 | 带图请求正常（不经过此 bug 路径） |
| 采样参数 | 调整 temperature/top_p/repetition_penalty 均无效 |
| HF eager | 同环境同模型完全正常 |

---

## 二、推理全流程

### 2.1 请求路径

```
用户浏览器
  → Vite 前端 (port 20941)
  → Demo 网关 (port 8000, server/app.py)
  → routers/chat.py: _chat_vlm()
  → adapters/vlm/moss_vl_sglang/adapter.py: SglangOfflinePool.generate_stream()
  → sglang sidecar (port 30800, sglang.launch_server)
```

### 2.2 sglang 内部处理链

```
/generate (HTTP)
  → TokenizerManager._tokenize_one_request()
    → obj.input_ids is not None → input_ids = obj.input_ids (11 tokens)
    → is_mossvl = True → should_run_mm_processor = True
    → mm_processor.process_mm_data_async(image_data=None, input_text=input_ids)
      → load_mm_data(): prompt = self._tokenizer.decode(input_ids)  // decode 回文本
      → process_mm_data(): processor(text=[text], padding=True)     // HF AutoProcessor 重新处理
      → _build_mm_items(): 检查 pixel_values                         // ★ bug 在这里
      → 返回 MultimodalProcessorOutput (input_ids=11, mm_items=[...])
    → input_ids = mm_inputs.input_ids (11 tokens)
  → Scheduler.handle_generate_request()
    → _get_multimodal_inputs(): MultimodalInputs.from_processor_output()
    → pad_input_ids_func(): model.pad_input_ids(input_ids, image_inputs)
      → _get_encoder_len(mm_inputs): 根据 mm_items[0].grid_thw 计算 encoder_len  // ★ 17
      → _build_encoder_prefix_pad_ids(): [704833207] * 17          // ★ 17 个 pad token
      → 返回 [704833207 × 17] + [11 个真实 token] = 28 tokens
    → req.origin_input_ids = 28 tokens
    → prompt_tokens = len(req.origin_input_ids) = 28
  → Prefill (28 tokens: 17 pad + 11 real)
    → KV cache 写入 28 个 token 的 K/V
  → Decode (step by step)
    → attention 读 28 个 KV (含 17 个垃圾 pad token)
    → EOS logit 从 step 7 开始系统性攀升
    → step 10: EOS 成为 top-1 → 早停
```

### 2.3 正常路径（应有）

```
/generate (HTTP)
  → TokenizerManager._tokenize_one_request()
    → input_ids = 11 tokens
    → mm_processor.process_mm_data_async()
      → _build_mm_items(): pixel_values is None OR input_ids 无 image_token_id → 返回 []
      → 返回 MultimodalProcessorOutput (input_ids=11, mm_items=[])
    → input_ids = 11 tokens
  → Scheduler.handle_generate_request()
    → pad_input_ids(): encoder_len = 0 (mm_items 为空) → 返回 11 tokens 不变
    → prompt_tokens = 11
  → Prefill (11 tokens)
  → Decode → 正常生成
```

---

## 三、问题产生的阶段与原因

### 3.1 产生阶段

**sglang 的 multimodal processor 阶段**（`_build_mm_items`），在 tokenizer_manager 和 scheduler 之间。

### 3.2 根本原因

**HF AutoProcessor 对纯文本输入也返回非 None 的 `pixel_values`**：

```python
# 实测：对纯文本 "<|im_start|>user\n介绍一下中国历史..." 调用 AutoProcessor
result = proc(text=[text], padding=True, return_tensors="pt")
# result["pixel_values"] → tensor(shape=[64, 768])  ← 非 None！是 placeholder
# result["grid_thw"]     → tensor([[1, 8, 8]])       ← 1帧 8×8 的"空图片"
```

AutoProcessor 的行为是：无论有无图片输入，都创建一个默认的 1×8×8 placeholder。这是 HF processor 的设计（不是 bug），但对下游 sglang 的 moss_vl 处理器造成了问题。

### 3.3 传导链

```
HF AutoProcessor 返回 placeholder pixel_values (非 None)
  → moss_vl._build_mm_items() 检测到 pixel_values is not None → 创建 image item
  → image item 携带 grid_thw=[[1,8,8]]
  → scheduler: _get_encoder_len() 根据 grid_thw 计算 encoder_len = 17
    (tokens_per_media = 8*8/4 = 16, +1 separator = 17)
  → pad_input_ids(): 在 11 个真实 token 前面加 17 个 pad token (704833207)
  → 实际 prefill 序列 = 28 tokens (17 pad + 11 real)
  → KV cache 中有 17 个垃圾 token 的 K/V
  → decode 的 attention 读到 17 个垃圾上下文
  → EOS logit 系统性攀升 (step 0: gap=9.75 → step 10: gap=0.0)
  → 早停 / 塌缩
```

### 3.4 EOS 攀升的实测数据

逐步对齐 sglang 和 HF 的 logprob，在翻转点（step 10）对比：

| step | sglang EOS 排名 | sglang gap (top1-EOS) | HF gap (top1-EOS) | 偏差 |
|---|---|---|---|---|
| 0 | >200 | >5.1 | 9.75 | ~4.6 |
| 7 | 第4 | 5.75 | 9.62 | 3.87 |
| 8 | 第2 | 0.375 | 9.25 | **8.88** |
| 9 | 第1 | 0.5 | 7.25 | **6.75** |
| 10 | **top1** | 0.0 | 5.25 | **5.25** |

偏差 5-9 的 logit 差远超 bf16 噪声水平（~0.08），证实是**系统性偏差**而非随机噪声。

---

## 四、定位过程

### 4.1 第一阶段：现象复现与环境隔离

- 直连 sglang `/generate` 复现问题（greedy 早停、ignore_eos 塌缩）
- HF eager 同 prompt 同参数完全正常 → 问题在 sglang 路径
- 多模态（带图）正常 → 问题仅在纯文本路径
- 采样参数调整无效 → 不是采样随机性问题

### 4.2 第二阶段：排除法排查

逐项排除可能的根因：

| 排除项 | 方法 | 结论 |
|---|---|---|
| 模型权重 | HF 直驱同权重同参数完全正常 | ✗ 排除 |
| chat template | 网关 apply_chat_template 与手拼模板逐字一致 | ✗ 排除 |
| 采样后端 | 去掉 `--sampling-backend ascend` 用默认 pytorch 采样，仍早停 | ✗ 排除 |
| attention 后端 | `--attention-backend torch_native` 替换 ascend，仍早停 | ✗ 排除 |
| mrope NPU kernel | 纯文本 positions 三轴相同，mrope 误差为 bf16 舍入级 | ✗ 排除 |
| 物理卡硬件 | 卡 7 和卡 15 新起实例均复现 | ✗ 排除 |
| 实例状态 | 新起实例也复现（非"运行久了劣化"） | ✗ 排除 |
| 并发合批 | 串行单请求也复现 | ✗ 排除 |
| cross-attention mask | 纯文本无 encoder，mask 逻辑不触发 | ✗ 排除 |

### 4.3 第三阶段：逐算子精度对比

使用 sglang 的 `--debug-tensor-dump-output-folder` 全层 dump + HF forward hook dump，逐算子 A/B 对比：

| 算子 | sglang vs fp32 参考 | HF vs fp32 参考 | 结论 |
|---|---|---|---|
| embedding 查表 | 逐位一致 | — | 无辜 |
| RMSNorm | 0.00084 | 0.00154 | sglang 更准 |
| GEMM (qkv_proj) | 0.000244 | 0.00195 | sglang 更准 |
| rope | triton/native/数学一致 | — | 无辜 |
| KV pool 写入 | 逐位一致 | — | 无辜 |
| attention kernel (sdpa) | cos 0.99999 (control3) | — | 无辜 |

**所有算子独立精度正常**。三方层间噪声对比（CPU-fp32 真值 vs HF-NPU-bf16 vs sglang-NPU-bf16）显示两条曲线几乎重合（L47 cos ~0.9996）。

### 4.4 第四阶段：fp32 数学展开实验

将 decode + extend 的 attention 都替换为 fp32 纯数学展开（matmul + fp32 softmax），**塌缩依旧**。排除 attention kernel 精度问题。

### 4.5 第五阶段：逐步 logprob 对齐（突破口）

让 sglang 和 HF 各自逐步生成 25 个 token，逐步对齐 logprob：

- **前 10 步两边逐 token 完全一致**（sg top1 = HF top1）
- **step 10 分歧**：sglang 选 EOS(151645)，HF 选 220(内容 token)
- **EOS logit 系统性攀升**（step 7-10 从第 4 名升到 top1）
- 偏差 5-9 的 logit 差远超噪声 → **系统性偏差，不是随机噪声**

### 4.6 第六阶段：定位 prompt_tokens 异常

发现 sglang `prompt_tokens=28` 而 HF 只有 `11`——**差 17 个 token**。

在 scheduler 加 debug 日志追踪：

```
DEBUG pad_input_ids: BEFORE len=11 ids=[151644, 872, 198, ...]
DEBUG image_inputs: mm_items=1 encoder_lens_cpu=None
DEBUG pad_input_ids: AFTER  len=28 ids=[704833207, 704833207, ...]
```

**根因确认**：`mm_items=1`（不应有）→ `_get_encoder_len` 计算 17 → `pad_input_ids` 加 17 个 pad token。

### 4.7 第七阶段：定位 mm_items 来源

```python
# mm_processor 独立调用测试
result = proc.process_mm_data_async(image_data=None, input_text=IDS, ...)
# result.mm_items = [1 item]  ← 不应为 1！
# item0: modality=IMAGE, feature=Tensor([64,768]), grid_thw=[[1,8,8]]
```

追踪到 `_build_mm_items`：

```python
pixel_values = processor_output.get("pixel_values")
if pixel_values is None:
    return []
# pixel_values 不是 None（HF AutoProcessor 的 placeholder）→ 继续创建 item
```

**HF AutoProcessor 对纯文本也返回非 None 的 pixel_values**（shape [64,768], grid_thw [1,8,8]）——这是 HF processor 的设计行为，但 sglang 的 moss_vl 处理器未对此做过滤。

---

## 五、修复

### 5.1 修复内容

**文件**：`sglang/python/sglang/srt/multimodal/processors/moss_vl.py`
**方法**：`_build_mm_items`

在 `pixel_values is not None` 检查之后，增加 input_ids 是否包含 image_token_id 的守卫：

```python
def _build_mm_items(self, processor_output, input_ids):
    pixel_values = processor_output.get("pixel_values")
    if pixel_values is None:
        return []

    # ★ 新增守卫：HF AutoProcessor 对纯文本也返回 placeholder pixel_values
    # 如果 input_ids 中不含 image_token_id，说明是纯文本请求，不应创建 image item
    if self.image_token_id is not None:
        ids_flat = input_ids.flatten().tolist() if isinstance(input_ids, torch.Tensor) else list(input_ids)
        if self.image_token_id not in ids_flat:
            return []

    item = MultimodalDataItem(modality=Modality.IMAGE, feature=pixel_values, ...)
    ...
```

### 5.2 同步修复：mrope NPU 位置编码（独立 bug）

**文件**：`sglang/python/sglang/srt/layers/rotary_embedding/mrope.py`
**方法**：`forward_npu`

原代码用 `torch_npu.npu_mrope` 硬编码 `mrope_section=[0,0,0]` 且忽略 `mrope_interleaved`，导致多模态请求的旋转位置编码损坏。修复为使用 triton 融合 kernel（`forward_triton`），正确处理 mrope_section 和 mrope_interleaved。

### 5.3 验证结果

| 指标 | 修复前 | 修复后 |
|---|---|---|
| prompt_tokens | 28 (11+17 pad) | **11** |
| greedy 300 token | 11 步早停 / "龟龟龟" 塌缩 | **完整 300 token 正常内容** |
| sample 300 token | 重复塌缩 | **完整流畅** |
| 吞吐 | 2-5 tok/s (早停) | **27 tok/s** |
| 多模态（带图） | 位置编码损坏 | **正常（mrope triton 修复）** |

---

## 六、代码改动清单

### 6.1 sglang fork（Moss-VL/MOSS-VL，npu/minimal 分支）

| 文件 | 改动 | commit |
|---|---|---|
| `sglang/python/sglang/srt/multimodal/processors/moss_vl.py` | `_build_mm_items` 增加 image_token_id 守卫 | `f1839fe` |
| `sglang/python/sglang/srt/layers/rotary_embedding/mrope.py` | `forward_npu` 改用 triton kernel | `f1839fe` |

### 6.2 Demo 仓库（MOSS-VL-Realtime_Demo）

| 分支 | 文件 | 改动 | 用途 |
|---|---|---|---|
| `fix/offline-hf-provider` | `server/adapters/registry.py` | 新增 hf offline provider | 止血回退 |
| `fix/offline-hf-provider` | `server/app.py` | lifespan provider 分发 | 止血回退 |
| `demo/full`（主分支） | 无改动 | 与原生一致 | — |

### 6.3 运行栈（Moss-vl-it3/MOSS-VL）

| 分支 | 改动 | commit |
|---|---|---|
| `deprecated/npu/torch210-compat` | 同步 sglang fork 的两个修复 | `538c10e` |

### 6.4 配置（.env.deploy，不在 git 跟踪）

```
OFFLINE_PROVIDER=sglang    # 主路径
OFFLINE_GPUS=7             # sglang sidecar 物理卡
```

止血切换：`OFFLINE_PROVIDER=hf` + `OFFLINE_GPUS=3`（需在 `fix/offline-hf-provider` 分支）。

---

## 七、社区关联

vllm-ascend #14663（2026-08-20，open）：Qwen3-VL-8B-Instruct 在 Ascend 910B3 + CANN 9.1.0 上中文输入产生乱码/重复/空输出，英文正常，采样参数无效。与我们的问题高度相似（mrope VL 模型 + 910B + 中文乱码重复）。该 issue 只有现象没有定位——我们的根因定位（placeholder pixel_values → pad 垃圾 token）可能是同一类问题的解。

---

## 八、经验教训

1. **"每个算子都精确" ≠ "端到端无系统性偏差"**：逐算子 A/B 能排除单点算子 bug，但不能发现上游输入被污染（17 个 pad token 混入）。必须做端到端同 step 对比才能定位。

2. **prompt_tokens 是关键信号**：sglang 返回的 `prompt_tokens=28` vs 预期 `11` 是最早的异常信号，应该在第一时间追踪。我们在排查后期才注意到这个数字。

3. **HF AutoProcessor 的隐式行为**：processor 对纯文本也返回 placeholder pixel_values——这不是 bug（HF 设计如此），但下游消费者（sglang）必须对此做防御。GPU 上不出问题是因为 GPU 的 sglang fork 可能走了不同的 processor 调用路径或有不同的过滤逻辑。

4. **排除法的天花板**：当所有算子都被排除后，问题出在"算子之前"（输入污染）——这个阶段不在逐算子验证的覆盖范围内。逐步 logprob 对齐是突破这种僵局的有效手段。
