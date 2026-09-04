# MOSS-VL Realtime 网关接入 Plan

> 依据：评审文档（acnc6zeentra wiki XswEwpphAikDp8kDeAzcnfVtnoc）+ 协议文档（fudan-nlp wiki YA6mwLu71iRsvaknpKrcNg9qnwh）网关适配方案章。
> 目标：改完后能逐条回答评审文档 §5/§6.2/§7/§8/§9/§14 的问题，P0 可验收。

## 0. 已定决策（2026-09-04 与负责人确认）

| 决策项 | 结论 |
|---|---|
| 网关厚度 | **薄网关**：客户侧鉴权、并发额度、限流、白名单/灰度全部留给平台；本层只做会话管理、路由、透传、计量、保活 |
| 多会话形态 | **单实例多会话**：`--max-running-requests` N>1，多会话共享单卡 KV 池；安全 N 由压测定（P5） |
| 向量 embedder | hashing（默认）与 BGE/ColBERT 都保留可选，**压测 A/B 对比后定** |
| memory 生命周期 | **绑定会话 + grace 窗口**：会话活跃期累积；断连进 grace 保留；grace 过期/会话 GC 时删除该会话 memory 行。无跨会话累积，存储有界 |
| omni 故障恢复 | **一期做**：复用 rollover re-seat 机制，新建 omni 会话 + memory prefix（摘要+facts+原文尾巴）prefill 接续 |
| 客户端断线重连 | 已实现（grace + replay 环形缓冲，上下文完整保留）；对评审口径：支持重连恢复 |
| 动态修正 | **不支持**：模型输出只增不改，omni 无对应事件；口径为「待模型侧确认输出语义，若支持走协议版本协商新增事件」 |
| 网关 HA | 一期单副本；gateway 重启 = 会话全断、客户端重连新建（语义明确） |

## 1. 架构定位

```
客户 → 平台网关（鉴权/额度/限流/白名单/Docs/Playground）
     → 本适配网关（REST 控制面 + WSS 数据面透传 + 会话路由 + 计量 + 保活 + 故障接力）
     → sglang-omni 推理实例池（内网明文，单实例多会话）
     + 4B decide/compact 后端（memory 子系统共享）
     + memory store（sqlite，会话级隔离，grace 绑定清理）
```

- 数据面协议 = 协议文档「MOSS-VL Realtime 服务协议（实现对齐版）」章，omni 已全量实现，网关**只做透传/整形/关闭策略，不改事件语义**。
- demo 前端现有链路（翻译协议 session_ws）不动；新增 `/v1/realtime/*` 对外端点，双模并存。

## 2. 实施阶段

### P1 网关核心（控制面 + 数据面透传）

新代码：`server/gateway/` 包（rest.py / ws.py / tokens.py / metrics 钩子），复用 `adapters/vlm/moss_vl_sglang_omni/pool.py` 做实例池。

- REST（内部入口，供平台调用）：
  - `POST /v1/realtime/sessions` → 返回 session_id、ws_token（一次性、60s）、WSS 地址
  - `GET /v1/realtime/sessions/{id}` → 状态（ready/streaming/parked/done）、累计帧数、当前 turn_id（网关在透传路径上观测事件流得出）
  - `POST /v1/realtime/sessions/{id}/reset` → 关旧数据面连接、同 session_id 发新 ws_token；不存在返回 session_not_found
  - `DELETE /v1/realtime/sessions/{id}` → 主动释放；数据面异常断连兜底释放
  - `GET /health` → 实例数、活跃会话数、容量水位（pool.status() 聚合）
  - `GET /v1/models` → 转发 omni
- WSS `/v1/realtime`：ws_token 校验（非法/过期 → error + close 1008）→ session_id 粘性路由到 pool 实例 → 事件流双向透传
- 保活：网关对客户端周期 WS ping/pong，无 pong 判定断连进 grace；对下游实例维持长连，客户端抖动不影响推理侧
- 错误码：网关侧 `ws_token_invalid / ws_token_expired / session_not_found`；后端透传 `session_capacity_exceeded(1013) / invalid_request / input_submission_failed / response_failed`；关闭码 1008/1009/1013（1009 帧过大需先实测 omni 行为，缺则网关补）
- 客户侧鉴权/额度/限流错误码（invalid_api_key、per_customer_quota_exceeded、rate_limited）**平台发，本层不实现**
- 验收：pytest 覆盖全部端点；ws 客户端脚本走通「建会话→configure→推帧→delta→silence→reset→销毁」闭环 + 「无 token / 过期 token / 错误 session_id / 容量满」四条异常路径

### P2 多会话与故障接力

- `deploy.conf` 暴露 `MAX_RUNNING_REQUESTS`（脚本已支持，默认 1 改为压测定值）
- pool 路由改为「实例内最少活跃会话优先」，容量 = 实例数 × max_running_requests
- KV 规划：单卡 KV 池 ÷ N = 每会话安全上下文，启动预检放不下即拒绝启动（omni 已有），运行期 KV 紧张按「最重会话中止」降级（omni 已有，透传 response_failed）
- **omni 故障接力**：pool 探活发现实例死 → 其上活跃会话触发 re-seat：RolloverManager.build_prefix()（摘要 + facts + 原文尾巴）→ 健康实例新建 omni 会话 → prefix prefill → 会话继续；prefix 未就绪回退 response_failed
- 验收：多会话并发用例（N 路同时推帧互不串台）；kill 实例演练（会话经 memory 接力恢复，语义上下文保留、原始帧历史丢失）

### P3 计量与日志

- 会话创建生成平台 trace id，记录与后端 request_id 映射
- 三维计量：会话时长（REST 建→销毁/断连）、输入帧数（统计 accepted）、输出 token（复用 `_TextTokenMirror` 本地 token 化）
- 对账格式落库
- 验收：一条会话跑完，日志记录字段完整可对账

### P4 监控告警

- 指标暴露：活跃会话数、实例水位、帧吞吐、delta 速率、错误码分布、异常断连率、4B decide/compact 排队深度
- 告警规则：实例不可用、水位超阈值、断连率突增
- 验收：指标可抓取；告警规则有触发测试记录

### P5 压测（回答 §7 + 定两个悬置参数）

- 多会话模拟客户端：N 路 WS 并发推帧 + 按频率提问
- 八维度全记录：硬件 / 视频输入（分辨率、编码、帧大小、FPS）/ 会话负载 / 并发（单实例、单卡、集群）/ 时延（首次有效响应、单轮，P50/P95/P99）/ 稳定性（成功率、断开率、队列积压、丢帧）/ 资源（GPU、显存、CPU、带宽）/ 容量策略
- 产出 1：**单实例安全 N**（max_running_requests 定值）
- 产出 2：**embedder A/B 对比**（真实会话数据，召回质量 + CPU 开销 → 定 hashing 还是 BGE）
- 产出 3：4B 共享瓶颈水位（N 路并发 compact/decide 的排队表现）
- 对外并发/SLA 只报满足时延与稳定性条件下的「稳定并发」
- 验收：报告覆盖 §7 表格全部维度

### P6 QA 验收用例（对应 §8）

- 8.1 功能：建连/推帧/随时提问/增量/静默/结束 → 已有链路 + 网关端点自动化
- 8.2 边界异常：非法格式、超大帧（1009）、seq 乱序、timestamp 回退、帧队列满背压、客户端异常断开（grace+replay 恢复）、park 超时、容量满、长时间无输入、kill 实例接力恢复
- 8.3 性能稳定：复用 P5 数据，每项含测试条件/目标值/实际值/是否通过
- 验收：逐条有对应自动化测试且通过

### P7 协议文档补齐（6.2 交付标准）

- B/C 每个事件补完整 JSON 示例 + 字段表（类型/必填/范围/默认值），写回飞书协议文档
- D 限制数值（分辨率/FPS/帧大小/时长）待 P5 压测后回填
- 补充定义：最大会话时长（目前只有 256K context 上限 + parked 超时，需显式定义）
- 断线重连、动态修正两项按 §0 口径写入待评审结论
- 验收：QA 能仅依文档写用例（对照文档手工接一次客户端验证）

### P8 部署与运维预案

- 拓扑：gateway 与 omni 实例同 GPU 节点；推理实例仅内网暴露；gateway 一期单副本
- 故障预案：实例宕机 → memory 接力（P2）；gateway 重启 → 会话全断、客户端重连新建
- 扩容：加卡加实例 → deploy.conf 变更 → 灰量验证
- 回滚：网关层独立可回滚，不动 omni
- 验收：预案文档 + kill 实例/重启 gateway 演练记录

## 3. 对评审文档的回答口径（改完后）

- 5.2：WS 正式协议 ✅；REST 先建 Session ✅（P1）；帧二进制 ✅；时间戳秒/float/必填/乱序拒绝 ✅；静默/增量/结束事件 ✅；动态修正=不支持待模型侧确认；心跳=网关 ping/pong ✅；断线重连=客户端侧支持（grace+replay）、实例故障 memory 接力 ✅（P2）；超限=拒绝+最重会话降级 ✅
- 6.2：A-D 结构 = 协议文档 + P7 补齐 JSON 示例
- 7：P5 压测报告
- 8：P6 用例
- 9.1：MaaS 注册字段全部可填；9.2：平台侧职责，我们提供协议与计量接口
- 14 API 后端：协议初版=协议文档 ✅；网关 WS/Session/并发控制=P1/P2 ✅；控制标记过滤=已有翻译层 ✅；重连/超时/超限=上述口径
- 14 推理研发：部署卡型/实例数=deploy.conf；压测计划=P5；稳定并发=P5 产出

## 4. 不做的事（与协议文档 §10 对齐）

- 客户侧鉴权/额度/限流/白名单（平台做）
- 跨会话 memory 复用（生命周期已定为会话级 + grace 清理）
- 动态修正（待模型侧确认）
- JoyAI 逐帧 HTTP 兼容层
