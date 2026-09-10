# SeedVR2 主模型 4x 放大链路改造方案

## Context（为什么要改）

用户用 `seedvr2_3b` / `seedvr2_7b` 作为**主模型**（非 upscaler）生成图像时，
当前实现是 **1x 输出**：

```
输入图 H×W → VAE encode 到 latent H/8×W/8 → 采样 → VAE decode 回 H×W
```

在 2080ti（11GB 显存）上，直接以高分辨率原图跑全链路既不省显存也慢。
用户期望改为 **4x 链路**：

```
输入图 H×W
  → 降采样到 ~H/4×W/4（宽度 clamp 在 [384, 512] 之间，高度等比）
  → SeedVR2 在该小图上恢复/生成（省显存、快）
  → 输出 4x 放大回目标分辨率 H×W
```

约束（用户明确指定）：

- 1/4 输入的**宽度必须 clamp 在 \[384, 512]**（低于 384 取 384，高于 512 取 512；高度按宽高比缩放）。

- 输出仍需精确回到请求的目标分辨率。

目标：更低显存峰值 + 更快耗时，同时输出分辨率保持用户指定。

## 现状事实（已探明）

- 主入口：`generateImageOnly(_ image: Tensor<FloatType>, ...)`
  `Libraries/LocalImageGenerator/Sources/LocalImageGenerator.swift:5089`

- 一切尺寸推导从 `image` 出发：

  - 整除断言 `precondition(image.shape[2] % (64 * imageScaleFactor) == 0)`（L5408-5409）

  - `startWidth = image.shape[2] / 8`（L5435）

  - `startHeight = image.shape[1] / 8`（L5436）

  - `imageScale = (startWidth/8, startHeight/8)`（L5468-5469）

- `colorCalibrationReference = image`（L5120）—— 后续 `upscaleImageAndToCPU` 需用**原全尺寸图**做色彩校准，不能替换。

- decode 后回程：L5064-5068 `upscaleImageAndToCPU(secondPassResult.0, colorCalibrationReference: colorCalibrationReference, ...)`。

- `expandImageForEncoding`（L3525-3535）对 seedvr2 直接返回 `(1, image, nil)`，通道/尺寸均由 `image` 决定，无额外扩帧。

## 实施方案

### 1. 降采样插入点

在 `generateImageOnly` 内，**L5120** **`colorCalibrationReference = image`** **之后、L5408 整除 precondition 之前**：

- `colorCalibrationReference = image`（原图引用）保持不变；

- 新增 `var workingImage = image`；

- 命中条件时（`modelVersion == .seedvr2_3b || .seedvr2_7b`）将 `workingImage` 置为降采样版；

- 把以下位置的 `image` 读为 `workingImage`：

  - L5408-5409 整除 precondition

  - L5435-5436 `startWidth/startHeight = image.shape[2]/8` → `workingImage.shape[2]/8`

  - L5632-5643 `firstPassImage = image` → `firstPassImage = workingImage`

- L5622 strength 早退分支**保持用原** **`image`**（该分支不做降采样生成，直接 faceRestore+upscaler，语义正确）。

### 2. 降采样数学（保留宽高比 + clamp + 对齐）

```swift
// 仅当 imageScaleFactor == 1 时启用（imageScaleFactor > 1 会破坏对齐约束）
ow = workingImage.shape[2]; oh = workingImage.shape[1]
destW = clamp(Int((Double(ow) / 4).rounded()), lower: 384, upper: 512)
destW = destW - destW % (64 * imageScaleFactor)   // 对齐 64，满足 L5408-09
destH = Int((Double(oh) * Double(destW) / Double(ow)).rounded())
destH = destH - destH % (64 * imageScaleFactor)
```

- **不用** `RealESRGANer.downscale`（它只支持固定整数因子的 averagePool，无法做 clamp 到任意宽度）。

- **改用**现成的 `Upsample(.bilinear, widthScale: destW/ow, heightScale: destH/oh)`（全库同款写法，L5636-5638、L1972 等），包在独立 `DynamicGraph()` 里做 `rawValue.toCPU()`，保持 `Tensor<FloatType>` 类型。

- 宽高比由 `destH = oh * destW / ow` 保持；clamp 只作用在 `destW` 上。

### 3. 4x 放大回程

在 L5064 `upscaleImageAndToCPU` 之前，对 seedvr2 命中分支：

1. decode 输出尺寸为 `destW×destH`（≈原宽/4）；
2. 先 `Upsample(.bilinear, widthScale: ow/decodeW, heightScale: oh/decodeH)` 精确放大到**原 H×W**（clamp 取整误差由此抹平，目标分辨率精确可控）；
3. 再交给现有 `upscaleImageAndToCPU(...)`：

   - 若 `configuration.upscaler` 已配置 → 仍在其后按需再放大；

   - 若未配置 → 返回精确原尺寸 H×W。
4. 非 seedvr2 分支不进入该降采样逻辑，原 1x 路径完全不变。

### 4. 落地范围（用户选定：最小硬编码，不加开关）

**不做配置开关**，硬编码仅对 seedvr2 主模型生效，启用条件：

```
modelVersion == .seedvr2_3b || .seedvr2_7b
  && imageScaleFactor == 1
  && 非 inpainting/editing modifier
  && 非视频多帧
```

所有尺寸推导仅在上述条件满足时改用 `workingImage`（降采样版），否则保持原 `image` 路径不变。

## 关键文件

- `Libraries/LocalImageGenerator/Sources/LocalImageGenerator.swift`（主改动：generateImageOnly 内降采样 + 放大回程）

- （仅当加配置开关时）`Libraries/DataModels/.../config_generated.swift` 或对应手写 Configuration 文件

## 风险与规避

| 风险                                      | 规避                                                               |
| --------------------------------------- | ---------------------------------------------------------------- |
| L5408-5409 整除断言失败                       | destW/destH 对齐到 `64*imageScaleFactor`；仅 `imageScaleFactor==1` 启用 |
| colorCalibration 错位                     | `colorCalibrationReference` 保持原全尺寸图（L5120 不变）                    |
| 放大回程尺寸不精确                               | 先精确 bilinear 到原 H×W 再走 upscaler，而非依赖 upscaler 的 4x 固定语义          |
| 1x 用户场景回归                               | 开关默认关闭，默认保持原路径                                                   |
| maskedImage / expandImageForEncoding 维度 | seedvr2 主模型非 inpaint，L5673 分支不进入，expand 直接返回原图，无影响               |

## 验证方式

1. 用 seedvr2 主模型 + 原图（如 1536×1024）生成：

   - diff/latent 阶段实际在 \~384-512 宽内跑，显存峰值显著低于 1x 现状；

   - 输出尺寸精确等于原 H×W；

   - 输出色彩正常（对照开关关闭的结果）。
2. 回归：非 seedvr2 模型 / seedvr2 1x（开关关）输出与改动前一致。
3. 在 2080ti 上对比耗时与显存：1x vs 4x 链路各一轮。

