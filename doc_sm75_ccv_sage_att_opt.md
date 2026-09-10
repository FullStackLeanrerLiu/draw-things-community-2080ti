***

## 一、可迁移代码资产总表

| 优化项                          | 来源项目                    | 可移植文件                                                                                      | 语言             | 依赖          | CCV 集成工作量       |
| :--------------------------- | :---------------------- | :----------------------------------------------------------------------------------------- | :------------- | :---------- | :-------------- |
| **INT8 QK + FP16 PV 融合内核**   | SageAttention-SM75-path | `csrc/qattn/attn_cuda_sm75.h`、`csrc/mma.cuh`                                               | 纯 CUDA + PTX   | 无           | 已内建（详见 2.1）     |
| **INT8 W8A8 GEMM（4 个 M 配置）** | vLLM                    | `csrc/cutlass_extensions/cutlass_w8a8/scaled_mm_c2x_sm75_dispatch.cuh`、`scaled_mm_c2x.cuh` | CUTLASS        | CUTLASS     | 中（适配 tensor 抽象） |
| **KV Cache INT8 量化**         | LMDeploy                | `src/turbomind/kernels/attention/attention_128_f16_sm75.cu`                                | 纯 CUDA         | CUTLASS 头文件 | 低               |
| **W4A16 Marlin**             | vLLM                    | `csrc/quantization/marlin/marlin.cu`、`marlin_mm.cuh`、`marlin_template.h`                   | 纯 CUDA         | CUTLASS     | 中               |
| **块稀疏注意力**                   | FlashInfer              | `include/flashinfer/attention/`（CUDA 模板）                                                   | CUDA + CUTLASS | CUTLASS     | 高（需适配调度接口）      |
| **INT8 MMA Atom**            | CUTLASS                 | `include/cutlass/arch/mma_sm75.h`（`SM75_8x8x16_S32S8S8S32_TN`）                             | CUTLASS 模板     | 无           | 低（已在使用）         |
| **软件流水线（替代 cp.async）**       | 自研                      | 双缓冲 + 手动预取                                                                                 | 纯 CUDA         | 无           | 中               |

***

## 二、可迁移代码的详细说明

### 2.1 SageAttention-SM75-path（INT8 QK + FP16 PV）✅ 已内建于 ccv，无需移植

**仓库**：`SageAttention-SM75-path`（**本工作区无此仓库**；且其内容已在 ccv 的
CUTLASS 内核中实现等效功能，见 `flash_fwd_int8_kernel.h` / `flash_int8_kernel_traits.h`）。

**核心实现要点（现状核对——均已内建于 ccv）**：

- INT8 QK 用 `mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32`（`SM75_8x8x16_S32S8S8S32_TN`）✅

- **手动** **`uint32_t`** **加载、禁用** **`ldmatrix`** ✅（`int8_cute_bypass_gemm` 已实现）

- int32/FP32 累加 ✅（INT8 QK 累加器为 `int32_t`；PV 走 FP16 tensor core）

- 在线 Softmax（warp 归约 + running max/denominator）✅（复用 `softmax_rescale_o`）

**⚠️ 本小节原始描述有一处错误，予以更正**：
原文称 "PV 必须使用 `mma.sync.aligned.m8n8k4.row.col.f16.f16.f16.f16`，FP16 m8n8k16 在 SM75 无效"。
**实际 ccv 的 FP16 PV 用的是** **`SM75_16x8x8`（m16n8k8）**（见
`flash_int8_kernel_traits.h` L9-24），QK 与 PV 是两个独立 TiledMma，PV 复用
原有 FP16 内核的 `m16n8k8`。**无 m8n8k4 需求**。此条为文档错误描述，不应据此移植。

**移植方式**：无需移植——功能已内建于 ccv。

***

### 2.2 vLLM INT8 W8A8 GEMM（4 个 M 配置）❌ 外部仓库资产，本工作区无

**仓库**：`vllm`

**可移植文件**：

- `csrc/cutlass_extensions/cutlass_w8a8/scaled_mm_c2x_sm75_dispatch.cuh`：4 个 Gemm 配置的调度逻辑

- `csrc/cutlass_extensions/cutlass_w8a8/scaled_mm_c2x.cuh`：Gemm 模板定义

**4 个配置的具体参数**：

| 配置                    | M 范围                 | ThreadblockShape | WarpShape | InstructionShape | 共享内存  |
| :-------------------- | :------------------- | :--------------- | :-------- | :--------------- | :---- |
| `sm75_config_M32`     | \[1, 32]             | 32×128×64        | 32×64×64  | 8×8×16           | 49152 |
| `sm75_config_M64`     | (32, 64]             | 64×128×128       | 32×64×64  | 8×8×16           | 49152 |
| `sm75_config_default` | (64, 128] 或 (256, ∞) | 128×128×64       | 64×64×64  | 8×8×16           | 32768 |
| `sm75_config_M256`    | (128, 256]           | 128×128×128      | 64×64×64  | 8×8×16           | 65536 |

**移植方式**：

1. 复制 `scaled_mm_c2x_sm75_dispatch.cuh` 和 `scaled_mm_c2x.cuh`。
2. 将 vLLM 的 `torch::Tensor` 参数替换为 CCV 的 `ccv_nnc_tensor_t*`。
3. 在 CCV 的 GEMM 调度中，根据 M 选择对应的配置。

**关键点**：所有配置均使用 `cutlass::arch::Sm75` 和 `InstructionShape = GemmShape<8, 8, 16>`。

***

### 2.3 LMDeploy KV Cache INT8 量化 ❌ 外部仓库资产，本工作区无（P3 记录项）

**仓库**：`lmdeploy`

**可移植文件**：

- `src/turbomind/kernels/attention/attention_128_f16_sm75.cu`：Turing 专用注意力内核

- `src/turbomind/kernels/attention/quantization.h`：量化/反量化内核

**核心实现**：

- per-head、per-token 非对称量化

- 量化：`scale = max(abs(K)) / 127`，`K_int8 = round(K / scale)`

- 反量化：`K_fp16 = K_int8 * scale`

**移植方式**：将量化/反量化内核提取为独立的 CCV 算子（`ccv_nnc_kv_quantize` / `ccv_nnc_kv_dequantize`），在 Attention 算子的预处理/后处理阶段调用。

***

### 2.4 vLLM Marlin（W4A16）❌ 外部仓库资产，本工作区无

**仓库**：`vllm`

**可移植文件**：

- `csrc/quantization/marlin/marlin.cu`：Marlin 内核入口

- `csrc/quantization/marlin/marlin_mm.cuh`：矩阵乘法实现

- `csrc/quantization/marlin/marlin_template.h`：模板定义

**核心实现**：

- W4A16：权重 4-bit 存储，激活 FP16

- GEMM 前将权重反量化回 FP16

- 不依赖 INT4 Tensor Core，通过减少显存带宽压力加速

**移植方式**：

1. 复制 Marlin 的 `.cu`/`.cuh` 文件。
2. 实现 W4A16 权重的加载和反量化逻辑。
3. 适配 CCV 的 tensor 抽象。

**注意**：vLLM PR #45375 将 `get_min_capability()` 从 80 降至 **75**，确认 Marlin 支持 SM75。

***

### 2.5 FlashInfer 块稀疏注意力 ❌ 外部仓库资产，本工作区无，且有 smem 溢出风险

**仓库**：`flashinfer`

**可移植文件**：

- `include/flashinfer/attention/` 下的 CUDA 模板

- 块稀疏注意力的调度逻辑

**核心实现**：

- 先算粗粒度（64×64）块级分数（FP16）

- 只对高分段执行完整 INT8 MMA

- 使用 `LDGSTS` 指令（128B 宽度）从稀疏 KV-Cache 加载数据

**移植方式**：

1. 提取块稀疏注意力的 CUDA 内核。
2. 适配 CCV 的调度接口——FlashInfer 使用 `paged_kv_t` 数据结构，CCV 需提供等效的 KV Cache 视图。
3. **注意共享内存上限**：Turing 的 64 KB/SM 限制需降低 tile 尺寸。

**风险**：FlashInfer 在 vLLM 中设置了 SM80 最低要求，因为 Turing 上存在 smem 溢出问题。CCV 实现时需特别小心。

***

### 2.6 CUTLASS INT8 Atom（✅ 已在使用）

**仓库**：`cutlass`

**可移植文件**：

- `include/cutlass/arch/mma_sm75.h`：`SM75_8x8x16_S32S8S8S32_TN` 定义

**核心定义**：

```cpp
template <>
struct Mma<gemm::GemmShape<8, 8, 16>, 32, int8_t, layout::RowMajor,
           int8_t, layout::ColumnMajor, int32_t, layout::RowMajor,
           arch::OpMultiplyAddSaturate> {
  // A: Array<int8_t, 4> → 1×uint32_t, RowMajor
  // B: Array<int8_t, 4> → 1×uint32_t, ColumnMajor
  // C: Array<int32_t, 2>, RowMajor
};
```

CCV 已经在使用这个 Atom，无需额外移植。

***

### 2.7 软件流水线（自研，无现成参考）

**原理**：SM75 无 `cp.async`，用双缓冲 + 手动预取部分隐藏内存延迟。

```cpp
int8_t* q_smem_buf[2];
int buf_idx = 0;
load_tile_to_smem(q_smem_buf[0], 0);
for (int k_tile = 0; k_tile < num_k_tiles; k_tile++) {
    if (k_tile + 1 < num_k_tiles)
        load_tile_to_smem_async(q_smem_buf[1 - buf_idx], k_tile + 1);
    compute_mma(q_smem_buf[buf_idx]);
    buf_idx = 1 - buf_idx;
}
```

**预期**：隐藏 **20-30%** 内存延迟。

***

## 三、可迁移内容与 Todolist 的对应关系

| Todolist 任务                       | 可迁移资产                             | 来源                      |
| :-------------------------------- | :-------------------------------- | :---------------------- |
| 修复 SageAttention SM75 fragment 加载 | `attn_cuda_sm75.h`、`mma.cuh`      | SageAttention-SM75-path |
| 实现 INT8 W8A8 GEMM                 | `scaled_mm_c2x_sm75_dispatch.cuh` | vLLM                    |
| 实现 KV Cache INT8 量化               | `attention_128_f16_sm75.cu`       | LMDeploy                |
| 移植 W4A16 Marlin                   | `marlin.cu`、`marlin_mm.cuh`       | vLLM                    |
| 集成块稀疏注意力                          | `include/flashinfer/attention/`   | FlashInfer              |
| 确认 INT8 MMA Atom                  | `mma_sm75.h`                      | CUTLASS                 |
| 软件流水线                             | 自研                                | —                       |

***

## 四、补全后的完整 Todolist

### P0：正确性修复与基础加速

- [x] **移植 SageAttention-SM75-path 的 INT8 QK + FP16 PV 内核**（✅ 已内建于 ccv CUTLASS 内核：
  `flash_fwd_int8_kernel.h`，手动 uint32\_t 加载、禁 ldmatrix；PV 为 m16n8k8 非 m8n8k4）
  - 无需移植——零接入。

- [x] **实现 INT8 W8A8 GEMM（4 个 M 配置）**（✅ 已以 `CCV_GEMM_INT8` 落地于
  `ccv_nnc_gemm_gpu_cublas.cu`：q6p/q8p → INT8 + per-col FP16 scale + cuBLAS INT8
  tensor-core GEMM + INT32 accum + scale epilogue；`CCV_GEMM_INT8_MIN_TILE` 门。默认关闭）
  - 说明：采用 cuBLAS INT8 路径，非 vLLM 4-M CUTLASS 模板；后续如需按 M 分档再对齐。

- [ ] **实现 KV Cache INT8 量化内核**（❌ LMDeploy 资产，外部仓库不在本工作区；P3 记录项）

- [ ] **集成 CUDA Graphs**（纯 CUDA Runtime API，无外部依赖）

- [x] **实现序列长度自适应路由**（✅ 已以 `CCV_QK_ROUTE=1` 落地：R*C≤128 → FP16，
  R*C>128 → INT8-QK；另有 `CCV_QK_MIN_TILE` 尺寸门。默认关闭）

- [ ] **实现 Qwen Image Edit 分时加载**

- [ ] **实现 SeedVR2 BlockSwap**（blocks\_to\_swap=16/24-36）

### P1：显存与精度优化

- [ ] **移植 W4A16 Marlin 内核**（❌ vLLM 资产，外部仓库不在本工作区）

- [x] **实现 Tiled VAE 编解码**（✅ 已落地：tile + cosine 软融合 + 强制高分 tile 触发，
  TiledDiffusion.swift / LocalImageGenerator.swift）

- [ ] **集成 NVML 显存检测**

- [x] **确认 FP32 累加**（✅ INT8-QK 累加器为 int32/FP32）

- [ ] **强制** **`-arch=sm_75`** **编译**

### P2：性能优化

- [ ] **手动软件流水线**（双缓冲 + 手动预取）

- [ ] **共享内存带宽优化**（PV 阶段用 `__shfl_sync` 替代 smem 往返）

- [ ] **移植 FlashInfer 块稀疏注意力**（❌ 外部仓库不在本工作区，且有 64KB/SM 限制风险）

- [ ] **量化融合**（Q/K 量化融合进数据加载阶段）

### P3：进阶优化

- [ ] **双累加器策略**（FP32 + FP16）

- [ ] **CUDA Graphs 多图捕获**（为 SeedVR2 每个窗口大小单独捕获）

- [ ] **MoE 专家级卸载**（如模型含 MoE 结构）

***

## 五、移植时的关键注意事项

1. **所有源文件都是纯 C++/CUDA 的**，Python 只是调度层。直接提取 `.cu`/`.cuh` 即可。
2. **适配 CCV 的 tensor 抽象是主要工作量**：`torch::Tensor` → `ccv_nnc_tensor_t*`，需要正确处理 stride、dtype、device 指针。
3. **CUTLASS 版本兼容性**：vLLM 和 FlashInfer 使用的 CUTLASS 版本可能与 CCV 不同，需检查 API 兼容性。
4. **共享内存上限**：Turing 的 64 KB/SM 限制会影响 FlashInfer 和 vLLM 配置的 tile 尺寸。
5. **`ldmatrix`** **禁用**：SageAttention-SM75-path 已修复此问题，移植时保持这一修复。
6. **FP16 PV 的 MMA 配置**：本条原始描述（"PV 必须用 `m8n8k4`，m8n8k16 在 SM75 无效"）为错误，已更正。实际 ccv 的 FP16 PV 用 `SM75_16x8x8`（**m16n8k8**），QK 与 PV 是独立 TiledMma，见 §2.1 更正说明。

***

**一句话总结**：所有需要的优化都有对应的纯 CUDA/CUTLASS 可移植资产。核心工作量在于**适配 CCV 的 tensor 抽象**，而非从零编写内核。优先移植 SageAttention-SM75-path（修复正确性）、vLLM 的 SM75 W8A8 配置（计算加速）、LMDeploy 的 KV 量化（显存减半），这三项覆盖了正确性、速度和显存三个维度。
