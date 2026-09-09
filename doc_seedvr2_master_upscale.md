# SeedVR2 作为主模型的 upscaler 链路（方案 A 落地）

## Context（为什么改）

`seedvr2_3b` / `seedvr2_7b` 在 draw-things 里作为**主模型**（非 upscaler 按钮）调用时，
其 spec 的 `modifier == .inpainting`（ModelZoo.swift L1076 等），本质是"成像到成像/全帧修复"模型。
项目中原实现是 **1x 输出**：`输入 H×W → VAE 8x → 采样 → 8x 解码回 H×W`，在 2080ti（11GB）上跑大分辨率既占显存也慢。

本次改造目标：把 seedvr2 当 **upscaler 主模型** 使用，让它执行"超分放大"语义。

### 交互前提（已确认）
- **seedvr2 的最终输出 = 结果分辨率**（调用方传入的 `image.shape`，即`targetSize`）。seedvr2 需要在结果分辨率上重建/放大，**不做后处理 bilinear 插值放大**。
- **中间尺寸只是条件**：喂给扩散的是一个缩小后的"中间图像"（workingImage，宽度 clamp 到 `[384,512]`），作为 seedvr2 的修复/重建条件；真正的高频细节由 seedvr2 在结果分辨率 latent 上重绘恢复。
- 中间尺寸由结果尺寸**反推计算**得到（见下）。

## 数据流（改造后）

```
调用方 image（结果分辨率 H×W，须 64 对齐）
   │  ── 计算中间尺寸 workingImage ──┐
   │  width: clamp(round(Hw/4), 384, 512)，对齐 64        │
   │  height: 等比，对齐 64                                │
   ▼
VAE encode workingImage  →  小 latent (~H/32 × W/32)
   │
   │  Upsample(bilinear) 上采样对齐到"结果尺寸 latent" (startH×startW)
   ▼
DiT 在【结果尺寸 latent】(startH×startW) 上做扩散/重绘   ←  关键：输出 latent = 结果尺寸
   │  条件来源 = 上采样后的小图 latent（含 mask 全 1 = 全帧修复）
   ▼
VAE decode  →  结果分辨率 H×W  == image.shape（精确，无后处理放大）
```

## 代码改动（已落地，LocalImageGenerator.swift · generateImageOnly）

| # | 行号 | 改动 |
|---|---|---|
| 1 | L5130-5132 | 守卫去掉 `.inpainting` 排除（seedvr2 主模型 modifier=inpainting，此前整条降采样对它失效），保留 `.editing` 排除 |
| 2 | L5435-5436 | 整除断言改回校验【结果尺寸】`image` 64 对齐 |
| 3 | L5462-5463 | `startWidth/startHeight = image.shape / 8`（回到结果尺寸）→ DiT 输出 latent 为结果尺寸 |
| 4 | L5708-5724 | inpainting 分支：`sample` 与 `encodedImage`（来自 workingImage 小图 latent）`Upsample` 上采样对齐到 `startWidth×startHeight`，作为结果尺寸条件注入源 |
| 5 | L6058-6070 | 删除 decode 后的 bilinear 放大回程（decode 已是结果尺寸，无需再放大） |

- 中间尺寸计算（保留，原实现）：`workingImage` = 结果尺寸/4、宽度 clamp `[384,512]`、等比、对齐 64 [L5130-5145]。
- `firstPassImage = workingImage`（L5670），仅作条件。
- 采样 `x_T`（L5843）、`conditionImage = maskedImage`（L5848）、`mask` 全 1（L5725+），三者在结果尺寸上自洽。
- `sample: nil`（L5846）：x_T 已含初始条件，无需再传 sample。

## 中间尺寸 = 解的部分（作为后续交互基线）

- **输出分辨率**：`image.shape`（结果分辨率）——seedvr2 输出的目标，不可变。
- **中间尺寸**：由输出分辨率反推，`width = clamp(round(W/4), 384, 512)` 对齐 64，`height` 等比对齐 64——仅作扩散条件，不参与输出尺寸。
- 若后续需要其他放大倍率/中间档，只需改中间尺寸的计算公式，链路接缝不变。

## 未实现（分阶段推进）

**DiT 去噪采样 latent 空间分块**（用户已选"先压测，分阶段"）：
- 现状：draw-things 的 `tiledDiffusion` 只作用于输入图/VAE encode（UNetFixedEncoder/FirstStage），**去噪 latent 空间分块是全新能力**，需在 DDIM/DPM++ 去噪主循环内实现"分块前向 + overlap 融合"。seedvr2 是单帧模型，latent 为 `NHWC(1,H,W,16)`，只需空间分块（无 temporal 需要）。
- 可复用：`TiledDiffusion.swift` L20-151 的 tile 网格/重叠权重/索引 helper（现用于 VAE 编解码，可重构复用）。
- 触发条件：压测结果显示存仍超 11GB 时再实现。

## 显存压测参数（11G 2080ti）

- `--weights-cache 8`（<9，预留系统余量；用户指定 11G 卡缓存放 9 以下）
- 组合：`--cpu-offload` + weights-cache 配额 + q6/q8 量化模型（seedvr2_3b q8p/q6p）
- 测例：结果尺寸 1024² / 2048²，`seedvr2_3b`，确认：不 OOM、输出分辨率=请求结果尺寸、显存峰值、连续请求稳定（配合已做模型复用缓存）。

## 验证清单

1. 输出分辨率 = 请求结果尺寸（无后放大偏差、无 H/W 反转）。
2. 无 OOM、显存峰值在 11GB 内。
3. 细节质量：seedvr2 在结果尺寸重绘，边缘/纹理比 bilinear 放大清晰（实测对比）。
4. 连续多请求稳定（weights-cache 命中，不走反复磁盘读/反量化）。

## 风险与备注

- `Upsample` 对 latent 的 `widthScale`→轴2 width、`heightScale`→轴1 height，语义与像素用法一致（已验证 ModelAddons.swift L1391-1404）。
- seedvr2 modifier 固定 `.inpainting` 且强制 `startStep>0`（L5413），必走 inpainting+resize 分支，无 strength 早退漏改。
- 本机因缺 Vendors submodule **无法编译验证**，改动需同步到 build box 编译 gRPC 镜像后压测。