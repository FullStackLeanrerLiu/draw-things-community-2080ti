# SM75（2080Ti 11GB）优化 Todolist

> 目标：让 seedvr2 / qwen\_image 等生成模型在 2080Ti 11GB 上**不崩坏、不 OOM、尽量利用 INT8 算力**。
> GPU: RTX 2080 Ti 11GB / CC 7.5 / 显存 11264 MiB。
> 关联文档：[doc\_seedvr2\_master\_upscale.md](./doc_seedvr2_master_upscale.md)

***

## 0. 复盘：当前 seedvr2 缩放链路的崩坏问题（已解决 / 结论）

### 技术要点（当前缩放数据流）

```
输入 image（结果分辨率 H×W，须 64 对齐）
  │ ── downscaleImageAndToGPU(scaleFactor=1) → 原图 H×W（scaleFactor=1 时不变）
  │ ── isSeedVR2DownscaleEnabled && seedVR2DownscaleWidth>0：
  │      destWidth  = clamp(round(Hw/4), targetWidth)，对齐 64    ← 384/512/768 可配
  │      destHeight = 等比，对齐 64
  ▼
  workingImage（小中间图 H'×W'）
  │
  ├─ [VAE encode] → 小 latent (~H'/8 × W'/8)
  │     │  Upsample(bilinear) 上采样回结果尺寸 latent (startH×startW)
  │     ▼
  ├─ [DiT] 在结果尺寸 latent(startH×startW) 上扩散/重绘   ← 输出 latent = 结果尺寸
  │     条件 = 上采样后的小图 latent（含 mask 全 1 = 全帧修复）
  │
  └─ ↓ VAE decode → 结果分辨率 H×W == image.shape（无后处理放大）
```

### 事实核查（关键）

- **seedvr2 文本编码**：`encodeSeedVR2`（TextEncoder.swift L3660）只读取提前算好的
  `positive_embedding` / `negative_embedding`（从 store 读 q6p/q8p 张量），**完全忽略输入图像 images**。
  → 它是**纯 DiT**；textImages 传什么图都不影响结果。**不是"要加载 qwen text 模型做视觉编码"**。
  **为什么还是加载 qwen text 模型**：seedvr2 checkpoint 里的 `positive_embedding`/`negative_embedding`
  正是用 Qwen 系列 text encoder 对（正/负）提示词离线算好后打包进模型的，运行期直接读回即可，
  **不再需要 live 跑 qwen**。若仍需 live 编码（自定义提示词），才需要 qwen text 权重；此时 qwen
  在 NNC 无 int8/int4 算子 → 只能 fp16 GEMM（见第 2 节）。

- **A/B/C/D 实验结论**：`--no-seedvr2-downscale`（D）画面正常；开启缩放（A/B/C）会在 SM75
  画面异常。定位：缩放链路里**小图 latent 被 bilinear 上采样回大 latent** 参与 DiT 重绘，
  该过程在 SM75 上的表现崩坏（学术上 bilinear 上采样 VAElatent 也通常不保真）。

***

## 1. 分层混合权重驻留（暂缓实现 · 记录技术路径）

> 讨论结论：方向正确，能带来热请求性能提升，但**不能简单把所有权重塞 shared memory**。

### 目标

在 `--cpu-offload` 下，让常用权重（text encoder、UNet 高频层）驻留 **CUDA managed
memory（共享显存）**，GPU 用时靠 page-fault 自动迁移，节点结束 back-out 保留，不 free。

### 怎么跑通（技术要点）

- **存储介质**：当前 `WeightsCache`（WeightsCache.swift L131-177）dGPU 路径用 `detach(.CPU)`
  / `attach(from:)` 做**全量 CPU↔GPU 拷贝**（PCIe 瓶颈）。目标改为承载在
  `cumallocmanaged`（ccv 的 cudaMallocManaged），免显式 copy。

- **不需要显式"挪入显存"**：managed memory 在 GPU kernel page-fault 时自动迁移所需页。

- **节点结束 back-out**：把 GPU 驻留页吐回 managed（不 free），下次请求直接复用。

### 分层规则（关键，避免 thrash）

| 层级   | 内容                                    | 策略                                                           |
| ---- | ------------------------------------- | ------------------------------------------------------------ |
| 常驻层  | 小/中尺寸 text encoder（qwen 等）+ UNet 高频层  | managed memory 常驻，可溢出到 RAM                                   |
| 大模型层 | 超显存模型（如 qwen2vl 7b fp16 ≈14GB > 11GB） | 保留 q8p 权重 + 按需 fp16 重建 + **块级 page prefetch**，避免 page thrash |
| 释放层  | 每次请求结束                                | GPU 驻留页 back-out，不 free                                      |

### 预期收益

- 消除每次全量 PCIe copy，热请求提速显著。

- 天然溢出，比硬性 CPU 拷贝更有弹性。

### 前提 / 风险

1. **Qwen2VL 无 int8/int4 算子**：text encoder 前向仍需全量 fp16 重建 → 省传输、省不掉重建。
2. **超显存模型整份压显存会 thrash**：必须块级驻留，不能 all-in。
3. 改动两层：Swift `WeightsCache` 介质 + ccv 层 `cumallocmanaged`。需构建机重编 ccv。

***

## 2. SM75 INT8 算力利用 + q6p/q8p 权重（高优先级）

> 现状痛点：**SM75 的 INT8 算力完全没用上**；q6p / q8p 量化模型被向上编译到 fp16，
> 既**浪费显存**（fp16 全量重建）又**影响性能**（没用上 INT8 tensor-core MMA）。

### 问题点

- q6p / q8p 权重量化符合 INT8 加速特性，理应走 INT8 GEMM（`CCV_GEMM_INT8` / `_ccv_nnc_gemm_int8`）。

- 但当前被**解压成 fp16 全量重建**后才算 → 显存翻倍 + 丢 INT8 吞吐。

- 尤其 text encoder（Qwen2VL）在 NNC 无 int8/int4 前向算子，只能 fp16 GEMM（记忆已确认）。

### 待办

- [ ] **查 ccv** **`CCV_GEMM_INT8`** **生效路径**：确认 q6p/q8p 权重在哪些模块能走 INT8 而非 fp16 重建。

- [ ] 给 seedvr2 / qwen\_image 的 **UNet 主模型**打通 INT8 GEMM（DiT 的 DenseMatMul），验证正确性 + 提速。

- [ ] **给 Qwen2VL text encoder 补 INT8/INT4 前向算子**（NNC 层），避免 fp16 全量重建；
  这是显存和 textEncode 双重瓶颈的核心。

- [ ] 用 INT8 后重测 A/B/C/D 缩放链路，看是否缓解画面异常 + 显存占用。

***

## 3. seedvr2 链路 vs ComfyUI（对比结论）

> 结论：当前方案整体方向与 ComfyUI 一致（结果尺寸重绘 + 小图条件），
> 差异在 tile 处理与显存策略。

### 当前方案

- DiT 在**结果尺寸 latent** 重绘；条件 = 上采样后的小图 latent（bilinear）+ mask 全 1。

- 高分时依赖 `tiledDecoding` / `tiledDiffusion`（FirstStage.swift）分块避免 OOM。

- 但：D 方案（禁用缩放）在高分辨率时 **tile 未生效 → OOM**；`--cpu-offload` 时共享显存未用 → 严重 OOM。

### ComfyUI 2080Ti 做法（参考）

- **Flash Attention**：注意力分块 + 内存复用，显存可省 50%+。

- **BlockSwap（动态模块管理）**：模型模块在 GPU/CPU 间按需调度，适配低显存。

- **VAE Tiling**：分块解码/编码降低峰值显存。

- **DiT 分块（tiled diffusion）**：在扩散去噪过程对 latent 空间分块，降峰值。

### 差异 / 待同步

- [ ] **高分 tile 生效**：修 tile 条件，确保 D 方案高分也分块解码/生成。

- [ ] **`--cpu-offload`** **用上共享显存**：利用 11GB 显存作为 offload 温点，而非全 CPU。

- [ ] 可评估引入 **BlockSwap** 作为 `--cpu-offload` 的替代/补充。

***

## 4. 其他已知待办

- [ ] `--weights-cache` 自动分配**按显存预留安全大小**（不只是 RAM）：读 GPU 总显存，
  收敛 `maxTotalCacheSize ≤ 显存 - 安全余量` + RAM 配额，二者取小。

- [ ] `--cpu-offload` 下优化 textEncode 热请求延迟（权重驻留 + 结构缓存复用）。

***

## 5. SageAttention SM75 调优（P1）

> **⚠️ 更正（新文档 doc_sm75_ccv_sage_att_opt.md 核对结论）**：本节早期描述基于 **Triton**
> 实现（Ph0rk0z fork / SageAttention-SM75-path）。但 **ccv 实际用 CUTLASS 实现**（flash_attn
> 下 `flash_fwd_int8_kernel.h` / `flash_int8_kernel_traits.h`），**不依赖 Triton** → 本节
> 所有 "Triton 版本锁定 / 强制 Triton 路径" 项对当前 ccv 路径 **不适用（N/A）**。对应结论见
> 新文档 §2.1 / §四；以下仅打勾已在 ccv CUTLASS 实现中落地的项。
>
> 现状：自定义 ccv 驱动（CUTLASS INT8 QK + FP16 PV 融合内核），SM75 上理论应比
> split-cross-attention 快（INT8 QK flops 提升约 2.3–2.4×）。实际收益与序列长度强相关，
> 小矩阵因固定量化开销反而更慢 -> 用 CCV_QK_ROUTE 做序列长度分档路由。
> 基准：1024 flops 27.2→52.1；2048 30.0→71.2；4096 27.9→68.1；8192 27.7→67.6（vs xformers FP16）
>
> - Wan2GP Turing 端到端 \~30% / MiniMax-H3 实测 16.8%（133.37s vs 160.25s）。

### P0 先确认"该快的场景真的快"（排除配置问题）

- [ ] **Triton 版本锁定 3.1.x / 3.2.x**（禁用 3.3+，否则内核编译失败或结果错误）。

- [ ] **量化粒度自适应**：小矩阵用 `per_block`，大矩阵用 `per_thread`。
  S < 256 → `per_block`；S ≥ 256 → `per_thread`。当前粒度选错会直接拖慢小矩阵。

- [ ] **强制 Triton 路径**：显式调 `sageattn_qk_int8_pv_fp16_triton()`，避免落到 CUDA 内核
  （Ph0rk0z fork 的 CUDA 内核结果错误，`mean_rtol/atol` 可 >1）。

- [ ] **累积器 FP32**：`pv_accum_dtype="fp32"`，FP16 累加长序列会溢出 → NaN。
  （ccv 已落地 ✅：INT8-QK 累加器为 int32/FP32，见新文档 §2.1）

- [ ] **Fragment 布局修复**：INT8 用手动 `uint32_t` 加载，**禁用** **`ldmatrix`**（`ldmatrix_m8n8x4`
  与 INT8 MMA 不兼容，实测数据错乱、画面崩坏）。用全 1 矩阵验证 fragment 布局正确性。
  （ccv 已落地 ✅：`int8_cute_bypass_gemm` 手动 uint32_t 加载、禁 ldmatrix，见新文档 §2.1）

- [ ] **Outlier Smoothing**：对 Q/K 做 mean 中心化，省略会导致量化误差放大。
  （⚠️ 本轮评估默认关闭，仅文档记录；若加会破坏现有输出，择机单独 A/B）

- [ ] **在线 Softmax**：K-tile 级在线 softmax + warp 归约（简单 dead-loop 会导致数值不稳）。

### P1 SM75 Tile 调优

| 优化项        | 具体动作                             | 预期效果                      |
| ---------- | -------------------------------- | ------------------------- |
| 小矩阵专用 tile | block\_M/N = 32×32（S ≤ 128）      | 降低固定开销占比（当前 32×32 小矩阵慢在此） |
| 大矩阵 tile   | block\_M/N = 64×64（S ≥ 512）      | 充分利用 INT8 MMA             |
| PV 独立 tile | PV 的 block 配置与 QK 解耦             | PV 不成为新瓶颈                 |
| K-tile 流水线 | 2-stage shared memory pipelining | 隐藏加载延迟                    |

### P2 移除反量化税

- [ ] 量化融合：Q/K 量化并进上一个内核，少一次内存读写。

- [ ] Scale / Smoothing 预计算：kernel 启动前算好，避免 per-tile 重复。

### P3 与 split-cross-attention 的序列长度路由

> ✅ 已落地（`CCV_QK_ROUTE`，默认关闭）：R*C ≤ 128 → 保持标准 FP16；R*C > 128 → 走
> INT8-QK。为当前 CUTLASS 实现下"小矩阵别硬开 INT8"的显式档位开关。见新文档 §2.1/§四。

| S             | 路径                              | 理由                 |
| ------------- | ------------------------------- | ------------------ |
| S ≤ 128       | **split-cross-attention**       | INT8 固定开销 > MMA 增益 |
| 128 < S ≤ 512 | **SageAttention (per\_block)**  | 量化开销可控             |
| S > 512       | **SageAttention (per\_thread)** | 大矩阵摊薄固定开销，收益最大     |

### Debug 流程

1. 入口（`sageattn()`）加日志：实际函数名 + Triton 版本 + `qk_quant_gran` + 输入 shape。
2. 官方 benchmark：`bench_baseline.py`（xformers）vs `bench_qk_int8_pv_fp16_triton.py`；
   flops 若远 <50 说明环境/内核编译有问题。
3. `nsys profile` 对比**量化/反量化内核** vs 注意力内核耗时；量化 + 反量化 > 注意力 30%
   → 量化为瓶颈。`ncu` 看 `sm__throughput.avg.pct_of_peak_sustained_elapsed` 与 occupancy。

### 2080 Ti 避坑清单

1. 不用 Triton 3.3+（编译失败/结果错）。
2. 不用 CUDA kernel 路径（结果错）。
3. 小矩阵别硬开 INT8 QK。
4. 未验证 fragment 布局前别信 INT8 输出。
5. 别省略 Outlier Smoothing。
6. 长序列必须 FP32 累加。

> 一句话：SM75 的 SageAttention 可用窗口 =「大序列 + per\_block/per\_thread 自适应量化 +
> Triton 3.1/3.2 + uint32\_t 手动加载 fragment + FP32 累加 + Outlier Smoothing」。
> 先 profile 定位，再针对性修复，勿盲目改 tile。

