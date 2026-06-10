# GraphColor — 手绘动漫图片主色提取算法

---

## 目录

1. [算法概述](#1-算法概述)
2. [推荐 Rust 依赖](#2-推荐-rust-依赖)
3. [核心数据结构](#3-核心数据结构)
4. [算法管线总览](#4-算法管线总览)
5. [预处理模块](#5-预处理模块)
   - 5.1 [透明通道合成](#51-透明通道合成)
   - 5.2 [等比缩放](#52-等比缩放)
   - 5.3 [BGR → Lab 色彩空间转换](#53-bgr--lab-色彩空间转换)
6. [分割模块](#6-分割模块)
   - 6.1 [Alpha 通道分割（最高优先级）](#61-alpha-通道分割最高优先级)
   - 6.2 [深度学习显著性分割](#62-深度学习显著性分割)
   - 6.3 [形态学后处理与连通域过滤](#63-形态学后处理与连通域过滤)
7. [聚类模块](#7-聚类模块)
   - 7.1 [加权特征缩放](#71-加权特征缩放)
   - 7.2 [K-Means 聚类](#72-k-means-聚类)
   - 7.3 [聚类中心还原](#73-聚类中心还原)
8. [评分模块](#8-评分模块)
   - 8.1 [像素占比因子](#81-像素占比因子)
   - 8.2 [亮度均匀性因子](#82-亮度均匀性因子)
   - 8.3 [空间中心距离因子](#83-空间中心距离因子)
   - 8.4 [视觉显著性因子](#84-视觉显著性因子)
   - 8.5 [综合评分](#85-综合评分)
9. [空洞检测与 GrabCut 精修](#9-空洞检测与-grabcut-精修)
   - 9.1 [空洞检测原理](#91-空洞检测原理)
   - 9.2 [空洞连通域过滤](#92-空洞连通域过滤)
   - 9.3 [GrabCut 精修设置](#93-grabcut-精修设置)
10. [区域分析完整流程](#10-区域分析完整流程)
11. [关键公式汇总](#11-关键公式汇总)
12. [配置参数一览](#12-配置参数一览)

---

## 1. 算法概述

GraphColor 从手绘动漫/插画风格图片中提取主体与背景的主色（Dominant Colors）。算法管线分为四个阶段：

```
预处理 → 分割 → 聚类 → 评分
```

**核心设计决策：**

- **色彩空间**：全部算法在 CIE Lab 空间进行，因为 Lab 是感知均匀的色彩空间，欧氏距离与人类感知色差近似线性相关
- **聚类策略**：对 Lab 三通道进行非对称加权，强化色度（a*/b*）差异，弱化亮度（L*）影响，使相近色相不同明度的颜色更容易归为一类
- **评分体系**：4 因子加权综合评分，同时考虑颜色的物理属性（占比、均匀性）和感知属性（空间位置、视觉显著性）
- **空洞处理**：利用背景主色检测主体内部穿透区域，再用 GrabCut 精修，解决环状主体的误识别问题

---

## 2. 推荐 Rust 依赖

| 功能 | 推荐 crate |
|------|-----------|
| 图像加载/缩放/色彩转换 | `image` 或 `opencv` |
| Lab 色彩空间转换 | `opencv` (cvtColor) 或自行实现 sRGB→XYZ→Lab |
| K-Means 聚类 | `linfa-clustering` 或 `kmeans` 或自行实现 |
| GrabCut | `opencv` (grabCut) |
| 形态学操作 | `opencv` 或 `imageproc` |
| 连通域分析 | `imageproc` (connected_components) |
| 显著性检测（深度学习分割） | **`rembg-rs`**（推荐，Python rembg 的 Rust 移植，基于 ONNX Runtime + U2-Net）或直接用 **`ort`**（ONNX Runtime Rust 绑定）加载 U2-Net/IS-Net ONNX 模型 |
| 线性代数 | `ndarray` |

---

## 3. 核心数据结构

```rust
/// 单张图片的分析结果
struct ImageResult {
    segment_method: String,
    foreground: RegionResult,
    background: RegionResult,
}

/// 单个区域（前景/背景）的主色分析
struct RegionResult {
    dominant_color: MainColor,    // 最高分主色
    main_colors: Vec<MainColor>,  // 按评分排序，最多 5 个
}

/// 单个主色
struct MainColor {
    lab: [f32; 3],       // L*[0,100], a*[-128,127], b*[-128,127]
    rgb: [u8; 3],        // RGB 0-255
    score: f32,          // 综合评分 0~1
    proportion: f32,     // 像素占比 0~1
}

/// 分割结果
struct SegmentResult {
    foreground_mask: BitVec,   // 前景掩码
    background_mask: BitVec,   // 背景掩码
    initial_mask: Option<BitVec>,  // 首次提取的 mask（用于空洞检测）
    foreground_ratio: f32,
}
```

---

## 4. 算法管线总览

```
输入图片 (BGR, 任意尺寸)
  │
  ▼
[1] 透明通道处理：若有 alpha，合成到白底并保留 alpha mask
  │
  ▼
[2] 等比缩放：最长边 ≤ max_size (默认 512)
  │
  ▼
[3] BGR → Lab 转换
  │
  ▼
[4] 主体提取 (按优先级)
  │   1. Alpha 通道分割（最可靠）
  │   2. 深度学习显著性分割（rembg / ONNX）
  │   3. 多策略融合分割（Fallback）
  │
  ▼
[5] 背景主色计算
  │   背景像素 → 聚类(k=6) → 评分 → 取 Top N 背景主色
  │
  ▼
[6] 空洞检测 + GrabCut 精修
  │   前景像素 → 对比背景主色 → 找空洞 → GrabCut 精修 mask
  │
  ▼
[7] 前景主色计算
  │   精修后前景像素 → 聚类(k=10) → 评分 → 输出主色
  │
  ▼
[8] 输出 ImageResult
```

**关键顺序说明**：先算背景主色，再用背景主色检测空洞，最后算前景主色。这是因为空洞检测需要可靠的背景颜色参考。

---

## 5. 预处理模块

### 5.1 透明通道合成

**输入**：RGBA 图片 $(w, h, 4)$
**输出**：RGB 图片（透明像素已合成到白底）+ 可选 alpha mask

$$\alpha = \frac{\text{img}_{[:,:,3]}}{255.0}$$

$$\text{bgr} = \text{bgr}_{\text{raw}} \cdot \alpha + 255.0 \cdot (1.0 - \alpha)$$

其中 $\text{bgr}_{\text{raw}} = \text{img}_{[:,:,:3]}$ 转为 float32，$\alpha \in [0, 1]$。

**Rust 实现注意**：对于 `image` crate，加载 PNG 时直接获取 `RgbaImage`，逐像素合成。

### 5.2 等比缩放

将图片最长边缩放到 `max_size`，保持宽高比。

$$\text{scale} = \frac{\text{max\_size}}{\max(h, w)}$$

$$\text{new\_w} = \text{round}(w \cdot \text{scale})$$

$$\text{new\_h} = \text{round}(h \cdot \text{scale})$$

使用双线性或 Lanczos 插值（`image` crate 默认双线性即可）。

### 5.3 BGR → Lab 色彩空间转换

OpenCV 的 `cvtColor(BGR2Lab)` 输出范围为 $[0, 255]$，需映射到标准 Lab：

$$L = \text{cv}_L \times \frac{100.0}{255.0} \quad \in [0, 100]$$

$$a = \text{cv}_a - 128.0 \quad \in [-128, 127]$$

$$b = \text{cv}_b - 128.0 \quad \in [-128, 127]$$

**Rust 实现注意**：若不使用 `opencv`，需自行实现 sRGB → XYZ → Lab 转换：

**sRGB → 线性 RGB**：对每个通道 $c \in \{R, G, B\} / 255.0$：

$$c_{\text{linear}} = \begin{cases} \left(\dfrac{c + 0.055}{1.055}\right)^{2.4} & c > 0.04045 \\[8pt] \dfrac{c}{12.92} & c \leq 0.04045 \end{cases}$$

**线性 RGB → XYZ**（sRGB D65）：

$$\begin{bmatrix} X \\ Y \\ Z \end{bmatrix} = \begin{bmatrix} 0.4124564 & 0.3575761 & 0.1804375 \\ 0.2126729 & 0.7151522 & 0.0721750 \\ 0.0193339 & 0.1191920 & 0.9503041 \end{bmatrix} \begin{bmatrix} R_{\text{linear}} \\ G_{\text{linear}} \\ B_{\text{linear}} \end{bmatrix}$$

**XYZ → Lab**（D65 参考白点 $X_n = 0.95047,\; Y_n = 1.00000,\; Z_n = 1.08883$）：

$$f(t) = \begin{cases} t^{1/3} & t > 0.008856 \\ 7.787 \cdot t + \dfrac{16}{116} & t \leq 0.008856 \end{cases}$$

$$L = 116 \cdot f\left(\frac{Y}{Y_n}\right) - 16$$

$$a = 500 \cdot \left[ f\left(\frac{X}{X_n}\right) - f\left(\frac{Y}{Y_n}\right) \right]$$

$$b = 200 \cdot \left[ f\left(\frac{Y}{Y_n}\right) - f\left(\frac{Z}{Z_n}\right) \right]$$

---

## 6. 分割模块

分割是算法的第一个关键步骤，目标是准确分离主体与背景。实现三种策略，按优先级回退。

### 6.1 Alpha 通道分割（最高优先级）

若图片有透明通道，直接使用。

$$\text{alpha\_foreground} = \alpha_{\text{mask}} > 16$$

有效性检查：

$$\text{ratio} = \frac{\text{alpha\_foreground 像素数}}{h \times w}$$

$$\text{if } 0.005 \leq \text{ratio} \leq 0.995 \text{：有效，直接返回}$$

阈值 16 的设定：$\alpha \leq 16$ 的像素视为完全透明，避免半透明边缘噪声。

### 6.2 深度学习显著性分割

使用 U2-Net 或 IS-Net 模型进行显著性检测。

**流程：**

1. 将 BGR 图片输入显著性模型
2. 模型输出 alpha matte（每个像素 0~255 的显著性值）
3. 阈值 > 16 作为前景

$$\text{neural\_mask} = \text{model\_output\_alpha} > 16$$

$$\text{ratio} = \frac{\text{neural\_mask 像素数}}{h \times w}$$

$$\text{if ratio} < 0.005 \text{ 或 ratio} > 0.995 \text{：提取失败，回退到 Fallback}$$

**Rust 实现方案**：通过 `ort` (ONNX Runtime) 加载预训练的 U2-Net/IS-Net ONNX 模型。输入为 resize 到模型要求尺寸（通常 1024×1024）的 RGB 图片，输出为单通道显著性图。

### 6.3 形态学后处理与连通域过滤

所有分割策略输出的 mask 都需要经过后处理。

#### 形态学操作

$$\text{mask} = \text{morph\_close}(\text{mask}, \text{kernel}=5\times5\text{椭圆}, \text{iter}=1)$$

$$\text{mask} = \text{morph\_open}(\text{mask}, \text{kernel}=3\times3\text{椭圆}, \text{iter}=1)$$

- **闭运算**：填充前景内部小孔洞
- **开运算**：去除前景外部小噪点

#### 连通域过滤

$$\text{min\_area} = \max(24,\; \text{total\_pixels} \times 0.002)$$

对每个连通域 $\text{label}$：

$$\text{area} = \text{连通域像素数}$$

$$\text{center\_score} = \text{连通域中心权重均值}$$

$$\begin{cases} \text{保留} & \text{if } \text{area} \geq \text{min\_area} \\ \text{保留} & \text{if } \text{area} \geq \dfrac{\text{min\_area}}{2} \text{ AND } \text{center\_score} > 0.45 \\ \text{丢弃} & \text{otherwise} \end{cases}$$

---

## 7. 聚类模块

### 7.1 加权特征缩放

聚类的核心创新：不直接使用原始 Lab 值，而是对三通道进行非对称加权缩放。

$$\text{scaled}_L = L \times w_{\text{lightness}}$$

$$\text{scaled}_a = a \times w_{\text{color}} \times a_{\text{boost}}$$

$$\text{scaled}_b = b \times w_{\text{color}}$$

**默认参数**：
- $w_{\text{lightness}} = 0.5$（亮度权重减半）
- $w_{\text{color}} = 1.0$（颜色基权重）
- $a_{\text{boost}} = 1.1$（a* 通道额外放大 10%）

**缩放后的距离度量**（等价于聚类中的欧氏距离）：

$$d^2 = (w_L \cdot \Delta L)^2 + (w_a \cdot \Delta a)^2 + (w_b \cdot \Delta b)^2$$

其中：
- $w_L = 0.5$
- $w_a = 1.1$
- $w_b = 1.0$

**设计意图**：
- 亮度权重最低 (0.5)：同一色相的不同明暗版本应归为同一类
- a* 通道最高 (1.1)：人类对红绿色差最敏感
- b* 通道居中 (1.0)：蓝黄次敏感

### 7.2 K-Means 聚类

使用 Mini-Batch K-Means 算法（大规模数据高效）。

- 关键参数：
  - `n_clusters`：前景 10，背景 6
  - `n_init = 3`：多次初始化取最优
  - `max_iter = 100`
  - `batch_size = 1024`（Mini-Batch 大小）

**聚类流程**：
1. 对 scaled_pixels 执行 K-Means
2. 获取每个像素的 cluster label
3. 统计每个聚类的像素数和中心
4. 将缩放后的中心还原为标准 Lab 值
5. 按像素数降序排序

**特殊情况处理**：当 $w_{\text{lightness}} = 0$ 时（完全忽略亮度），缩放后的 $L = 0$，聚类中心丢失 L* 信息，需从原始像素回填：

$$\text{center}_L[c] = \text{mean}(\text{original}_L[\text{pixels belonging to } c])$$

### 7.3 聚类中心还原

$$\text{center}_L = \frac{\text{scaled\_center}_L}{w_{\text{lightness}}}$$

$$\text{center}_a = \frac{\text{scaled\_center}_a}{w_{\text{color}} \cdot a_{\text{boost}}}$$

$$\text{center}_b = \frac{\text{scaled\_center}_b}{w_{\text{color}}}$$

当 $w_{\text{lightness}} = 0$ 时，$\text{center}_L$ 从原始像素回填（见上）。

---

## 8. 评分模块

聚类完成后，每个聚类（即每种颜色）需要通过多因子综合评分来确定其"主色"地位。

### 8.1 像素占比因子

最简单直接的因子：覆盖面积越大，越可能是主色。

$$\text{count\_score} = \frac{n_c}{N} = \text{proportion}$$

范围：$[0, 1]$

### 8.2 亮度均匀性因子

亮度越均匀的颜色越"纯"，越应该被选为主色。

每个聚类 $c$ 的亮度方差：

$$\text{var}_c = \frac{\sum\limits_{i \in c} (L_i - \text{mean}_{L_c})^2}{n_c}$$

所有聚类的平均方差：

$$\text{mean\_var} = \frac{1}{k} \sum_{c=1}^{k} \text{var}_c$$

指数衰减反转（方差越小分越高）：

$$\text{variance\_score}_c = \exp\left(-\frac{\text{var}_c}{\text{mean\_var}}\right)$$

**高效批量计算**（避免二次遍历）：

$$\text{sum}_{L_c} = \sum_{i \in c} L_i$$

$$\text{sum}_{L^2_c} = \sum_{i \in c} L_i^2$$

$$\text{mean}_{L_c} = \frac{\text{sum}_{L_c}}{n_c}$$

$$\text{var}_c = \frac{\text{sum}_{L^2_c} - \dfrac{\text{sum}_{L_c}^2}{n_c}}{n_c}$$

**Rust 实现注意**：遍历像素按 label 分组累加，或使用 `ndarray` 的 mask 操作。

### 8.3 空间中心距离因子

越靠近图像中心的颜色，越可能是主体颜色。

图像几何中心：

$$\text{center}_y = \frac{h}{2.0}, \quad \text{center}_x = \frac{w}{2.0}$$

$$\text{max\_dist} = \sqrt{\text{center}_y^2 + \text{center}_x^2}$$

每个像素到中心的距离：

$$\text{dist}_i = \sqrt{(y_i - \text{center}_y)^2 + (x_i - \text{center}_x)^2}$$

每个聚类的平均距离：

$$\text{avg\_dist}_c = \frac{\sum\limits_{i \in c} \text{dist}_i}{n_c}$$

归一化为接近度（越近越高）：

$$\text{center\_score}_c = 1.0 - \frac{\text{avg\_dist}_c}{\text{max\_dist}}$$

$$\text{center\_score}_c = \text{clip}(\text{center\_score}_c,\; 0,\; 1)$$

### 8.4 视觉显著性因子

最复杂的因子，统一整合色度与亮度的感知显著性。

#### 步骤 1：加权色度

$$\text{chroma} = \sqrt{w_{\text{a\_salience}} \cdot a^2 + b^2}$$

其中 $w_{\text{a\_salience}} = 1.3$（红绿轴强化）。

#### 步骤 2：归一化到 $[-1, 1]$

对整个区域的 $L^*$ 和 $\text{chroma}$ 计算均值和标准差：

$$\mu_L = \text{mean}(L), \quad \sigma_L = \text{std}(L)$$

$$\mu_{\text{chroma}} = \text{mean}(\text{chroma}), \quad \sigma_{\text{chroma}} = \text{std}(\text{chroma})$$

对每个聚类中心归一化：

$$L' = \text{clip}\left(\frac{L_{\text{center}} - \mu_L}{\sigma_L} \times 0.5,\; -1,\; 1\right)$$

$$\text{chroma}' = \text{clip}\left(\frac{\text{chroma}_{\text{center}} - \mu_{\text{chroma}}}{\sigma_{\text{chroma}}} \times 0.5,\; -1,\; 1\right)$$

**为什么要乘以 0.5**：将归一化后的标准差控制在 0.5 左右，确保大部分值落在 $[-1, 1]$ 内。clip 只截断极端异常值。

#### 步骤 3：色度/亮度权重分配

使用 sigmoid 函数从 logit 值计算权重：

$$w_{\text{chroma}} = \sigma(\text{chroma\_logit}) = \frac{1}{1 + \exp(-\text{chroma\_logit})}$$

$$w_L = 1 - w_{\text{chroma}}$$

默认 $\text{chroma\_logit} = 2.0$：

$$w_{\text{chroma}} = \frac{1}{1 + e^{-2}} \approx 0.881$$

$$w_L = 1 - 0.881 \approx 0.119$$

即色度权重约 88%，亮度权重约 12%。

#### 步骤 4：单项评分

$$\text{score}_L = (L')^2 \quad \text{（极端亮度得分高）}$$

$$\text{score}_{\text{chroma}} = \frac{\text{chroma}' + 1}{2} \quad \text{（映射到 [0, 1]，高色度得分高）}$$

#### 步骤 5：融合

$$\text{salience\_score} = w_{\text{chroma}} \cdot \text{score}_{\text{chroma}} + w_L \cdot (1 - \text{score}_{\text{chroma}}) \cdot \text{score}_L$$

**融合公式的设计意图**：

| 情况 | $\text{score}_{\text{chroma}}$ | $\text{score}_L$ | $\text{salience\_score}$ | 说明 |
|------|-------------|---------|---------------|------|
| 高色度，任意亮度 | 接近 1 | 任意 | 接近 $w_{\text{chroma}}$ | 鲜艳颜色直接得高分 |
| 低色度，极端亮度 | 接近 0 | 接近 1 | 接近 $w_L$ | 纯黑/纯白也能得一定分 |
| 低色度，中间亮度 | 接近 0 | 接近 0 | 接近 0 | 灰色得分最低 |

排序：**鲜艳色 > 极端明暗 > 中间灰**

### 8.5 综合评分

**原始权重**：
- $w_{\text{count}} = 0.5$（像素占比）
- $w_{\text{variance}} = 0.10$（亮度均匀性）
- $w_{\text{center}} = 0.1$（空间中心距离）
- $w_{\text{chroma}} = 0.3$（视觉显著性）

**归一化**（总和 = 1.0）：

$$w_{\text{count}} = \frac{0.5}{1.0} \approx 0.476$$

$$w_{\text{variance}} = \frac{0.10}{1.0} \approx 0.095$$

$$w_{\text{center}} = \frac{0.1}{1.0} \approx 0.095$$

$$w_{\text{salience}} = \frac{0.3}{1.0} \approx 0.286$$

**综合得分**：

$$\text{final\_score} = \text{count} \times 0.476 + \text{variance} \times 0.095 + \text{center} \times 0.095 + \text{salience} \times 0.286$$

**过滤**：$\text{final\_score} < 0.05$ 的聚类不输出。

---

## 9. 空洞检测与 GrabCut 精修

这是处理环状主体（如人物张开手臂、头发遮住脸部形成的空洞）的核心步骤。

### 9.1 空洞检测原理

**核心观察**：环状主体内部的"空洞"区域，其颜色与背景高度相似。

输入：
- $\text{fg\_pixels}$：前景区域内的像素 BGR 值 $(N, 3)$
- $\text{bg\_colors}$：Top K 个背景主色的 BGR 值 $(K, 3)$

计算每个前景像素到最近背景主色的距离：

$$\text{dist}(p, C_{\text{bg}}) = \min_{k} \left\| p_{\text{bgr}} - C_{\text{bg}_k, \text{bgr}} \right\|_2$$

$$\text{hole} = \text{dist}(p, C_{\text{bg}}) \leq 15.0$$

**默认阈值**：$\text{hole\_distance\_threshold} = 15.0$（BGR 欧氏距离）

**为什么用 BGR 而非 Lab**：
- 空洞区域的像素本身就是背景颜色，只是在主体的"内部"
- BGR 欧氏距离 < 15 表示颜色几乎相同
- 使用 BGR 避免额外的 Lab 转换开销

### 9.2 空洞连通域过滤

空洞候选区域需要进一步过滤，排除与图像边界相连的区域（那些是真正的背景，不是空洞）。

1. 将空洞候选映射回全图 mask
2. 连通域分析
3. 标记所有接触图像四边的连通域：

$$\text{border\_labels} = \text{unique}(\text{labels}_{[0, :]}) \cup \text{unique}(\text{labels}_{[-1, :]}) \cup \text{unique}(\text{labels}_{[:, 0]}) \cup \text{unique}(\text{labels}_{[:, -1]})$$

4. 仅保留不接触边界的连通域：

$$\text{internal\_holes} = \{ \text{label} \mid \text{label} \notin \text{border\_labels} \}$$

### 9.3 GrabCut 精修设置

初始化 GrabCut mask：
- 全部填充 $\text{GC\_PR\_BGD}$（可能背景）
- 空洞区域 $\rightarrow$ $\text{GC\_BGD}$（确定背景）
- 前景（非空洞）$\rightarrow$ $\text{GC\_PR\_FGD}$（可能前景）

设置强前景种子（前景主体的中心区域）：对前景的每个非边界连通域，若 $\text{area} \geq h \times w \times \text{hole\_min\_area\_ratio}$：

$$\text{cy}, \text{cx} = \text{连通域质心}$$

$$\text{ry}, \text{rx} = \sqrt{\text{area}} \times 0.45$$

在连通域中心画椭圆区域：

$$\text{gc\_mask}[\text{neural\_mask} \cap \text{椭圆区域}] = \text{GC\_FGD}$$

执行 GrabCut：

$$\text{cv2.grabCut}(\text{bgr}, \text{gc\_mask}, \text{iterations}, \text{GC\_INIT\_WITH\_MASK})$$

最终 mask：

$$\text{foreground} = (\text{gc\_mask} == \text{GC\_FGD}) \lor (\text{gc\_mask} == \text{GC\_PR\_FGD})$$

$$\text{foreground} = \text{foreground} \cap \text{neural\_mask}$$

**关键设计决策**：
- $\text{foreground} = \text{foreground} \cap \text{neural\_mask}$：GrabCut 结果不能超出深度学习初始 mask，避免 GrabCut "膨胀"到不相关区域
- 强种子只设在中心区域：边缘区域可能是背景，不能设为强前景

---

## 10. 区域分析完整流程

对任意区域（前景或背景）的像素，执行以下标准流程提取主色：

```
输入: lab_pixels (N, 3), pixel_coords (N, 2), 图像尺寸 (h, w)
  │
  ▼
[1] 计算区域统计量
  │   μ_L, σ_L, μ_chroma, σ_chroma
  │
  ▼
[2] 执行聚类
  │   cluster_result = KMeans(scaled_lab_pixels, k)
  │
  ▼
[3] 对每个聚类计算四因子评分
  │   - count_score = proportion
  │   - variance_score = exp(-var / mean_var)
  │   - center_score = 1 - avg_dist / max_dist
  │   - salience_score = 视觉显著性融合公式
  │
  ▼
[4] 综合评分 = Σ(factor_i * weight_i)
  │
  ▼
[5] 过滤 score < min_score_display 的聚类
  │
  ▼
[6] 按 score 降序排序
  │
  ▼
[7] 输出 Top 5 主色
  │   每个主色: Lab → RGB → Hex, score, proportion
```

---

## 11. 关键公式汇总

### 11.1 色彩空间转换

**BGR → Lab（标准范围）**：

$$L = \text{cv}_L \times \frac{100}{255}, \quad a = \text{cv}_a - 128, \quad b = \text{cv}_b - 128$$

**Lab → BGR**：

$$\text{cv}_L = L \times \frac{255}{100}, \quad \text{cv}_a = a + 128, \quad \text{cv}_b = b + 128$$

$$\text{cv\_lab} = \text{clip}([\text{cv}_L, \text{cv}_a, \text{cv}_b],\; 0,\; 255)$$

$$\text{bgr} = \text{cvtColor}(\text{cv\_lab}, \text{Lab2BGR})$$

### 11.2 色度（Chroma）

**聚类用色度**：

$$C^* = \sqrt{a^2 + b^2}$$

**视觉显著性用加权色度**：

$$\text{chroma} = \sqrt{w_{\text{a\_salience}} \cdot a^2 + b^2} \quad \text{其中 } w_{\text{a\_salience}} = 1.3$$

### 11.3 聚类特征缩放

$$d^2 = (w_L \cdot \Delta L)^2 + (w_a \cdot \Delta a)^2 + (w_b \cdot \Delta b)^2$$

$$w_L = 0.5, \quad w_a = 1.1, \quad w_b = 1.0$$

### 11.4 评分四因子

$$\text{count\_score} = \frac{n_c}{N}$$

$$\text{variance\_score} = \exp\left(-\frac{\text{var}_c}{\text{mean\_var}}\right), \quad \text{var}_c = \frac{\sum\limits_{i \in c} (L_i - \text{mean}_{L_c})^2}{n_c}$$

$$\text{center\_score} = \text{clip}\left(1 - \frac{\text{avg\_dist}_c}{\text{max\_dist}},\; 0,\; 1\right)$$

$$\text{avg\_dist}_c = \frac{\sum\limits_{i \in c} \sqrt{(y_i - h/2)^2 + (x_i - w/2)^2}}{n_c}, \quad \text{max\_dist} = \sqrt{(h/2)^2 + (w/2)^2}$$

$$\text{salience\_score} = w_{\text{chroma}} \cdot \text{score}_{\text{chroma}} + w_L \cdot (1 - \text{score}_{\text{chroma}}) \cdot \text{score}_L$$

其中：

$$\text{chroma} = \sqrt{w_{\text{a\_salience}} \cdot a^2 + b^2}$$

$$L' = \text{clip}\left(\frac{L - \mu_L}{\sigma_L} \times 0.5,\; -1,\; 1\right)$$

$$\text{chroma}' = \text{clip}\left(\frac{\text{chroma} - \mu_{\text{chroma}}}{\sigma_{\text{chroma}}} \times 0.5,\; -1,\; 1\right)$$

$$w_{\text{chroma}} = \frac{1}{1 + \exp(-\text{chroma\_logit})} \approx 0.881 \; (\text{logit}=2.0)$$

$$w_L = 1 - w_{\text{chroma}} \approx 0.119$$

$$\text{score}_L = (L')^2$$

$$\text{score}_{\text{chroma}} = \frac{\text{chroma}' + 1}{2}$$

### 11.5 综合评分

$$\text{final\_score} = \text{count} \times 0.476 + \text{variance} \times 0.095 + \text{center} \times 0.095 + \text{salience} \times 0.286$$

### 11.6 空洞检测距离

$$\text{dist}(p, C_{\text{bg}}) = \min_{k} \left\| p_{\text{bgr}} - C_{\text{bg}_k, \text{bgr}} \right\|_2$$

$$\text{hole} = \text{dist}(p, C_{\text{bg}}) \leq 15.0$$

### 11.7 背景相似度距离（Fallback 分割用）

$$\Delta E = \sqrt{((L - L_{\text{border}}) \times 0.45)^2 + ((a - a_{\text{border}}) \times 1.10)^2 + ((b - b_{\text{border}}) \times 1.10)^2}$$

### 11.8 中心权重

$$\text{dist} = \sqrt{(y - h/2)^2 + (x - w/2)^2}$$

$$\text{max\_dist} = \sqrt{(h/2)^2 + (w/2)^2}$$

$$\text{center\_weight} = \text{clip}\left(1 - \frac{\text{dist}}{\text{max\_dist}},\; 0,\; 1\right)$$

### 11.9 Alpha 合成

$$\text{bgr} = \text{bgr}_{\text{raw}} \times \frac{\alpha}{255} + 255 \times \left(1 - \frac{\alpha}{255}\right)$$

---

## 12. 配置参数一览

### 默认配置

| 参数 | 默认值 | 所属模块 | 说明 |
|------|--------|---------|------|
| `max_size` | 512 | 预处理 | 缩放后最长边 |
| `n_clusters_foreground` | 10 | 聚类 | 前景聚类数 |
| `n_clusters_background` | 6 | 聚类 | 背景聚类数 |
| `color_weight` | 1.0 | 聚类 | 颜色通道基权重 |
| `lightness_weight` | 0.5 | 聚类 | 亮度通道权重 |
| `a_boost` | 1.1 | 聚类 | a* 通道放大倍数 |
| `weight_count` | 0.5 | 评分 | 像素占比权重 |
| `weight_variance` | 0.10 | 评分 | 亮度均匀性权重 |
| `weight_center` | 0.1 | 评分 | 空间中心权重 |
| `weight_chroma` | 0.3 | 评分 | 视觉显著性权重 |
| `min_score_display` | 0.05 | 评分 | 最低输出分数 |
| `salience_a_weight` | 1.3 | 评分 | 显著性色度 a* 加权 |
| `salience_chroma_logit` | 2.0 | 评分 | 显著性色度/亮度 logit |
| `bg_colors_n` | 1 | 空洞检测 | 用于空洞检测的背景主色数 |
| `hole_distance_threshold` | 15.0 | 空洞检测 | 空洞 BGR 距离阈值 |
| `hole_min_area_ratio` | 0.002 | 空洞检测 | 空洞最小面积比 |
| `grabcut_iterations` | 1 | 空洞检测 | GrabCut 迭代次数 |
| `chroma_threshold` | 10.0 | 分割(Fallback) | 色度阈值 |
| `min_contour_area_ratio` | 0.02 | 分割(Fallback) | 最小轮廓面积比 |
| `foreground_score_threshold` | 0.5 | 分割(Fallback) | 前景得分阈值 |

### 关键参数调优建议

| 想达到的效果 | 调整参数 | 方向 |
|------------|---------|------|
| 聚类更关注色相，忽略明暗 | `lightness_weight` | 降低（0.2 或 0） |
| 红绿色差分辨率更高 | `a_boost` | 提高（1.2~1.5） |
| 主色数量更少但更精确 | `n_clusters_*` | 降低 |
| 更激进的空洞检测 | `hole_distance_threshold` | 提高（20~25） |
| 更保守的空洞检测 | `hole_distance_threshold` | 降低（10~12） |
| 评分更看重面积覆盖 | `weight_count` | 提高 |
| 评分更看重颜色鲜艳度 | `weight_chroma` | 提高 |
| 评分更看重颜色纯净度 | `weight_variance` | 提高 |

---

## 附录：Rust 实现注意事项

### 性能优化

1. **使用 `ndarray` 进行向量化计算**：避免逐像素 for 循环，利用 SIMD
2. **Mini-Batch K-Means**：大图片（>10 万像素）时，不必使用全部像素聚类，随机采样即可
3. **预分配缓冲区**：形态学、连通域分析等操作需要临时 buffer，提前分配避免频繁分配
4. **并行化**：前景和背景的聚类+评分可以并行处理

### 精度注意

1. **Lab 转换精度**：使用 `f32` 或 `f64` 存储 Lab 值，避免 uint8 精度损失
2. **聚类中心还原**：注意除法精度，尤其是 `lightness_weight` 接近 0 时
3. **sigmoid 数值稳定性**：`logit` 很大时使用 `1.0 - exp(-logit)` 近似

### 模型依赖

- 深度学习分割器需要 ONNX 模型文件（U2-Net 或 IS-Net），通常 100~176MB
- 推荐使用 **`rembg-rs`** crate，它是 Python rembg 的完整 Rust 移植，基于 `ort` (ONNX Runtime) + U2-Net/IS-Net，开箱即用
- 如果需要更细粒度控制，也可以直接用 **`ort`** crate 自行加载 ONNX 模型
- 建议作为可选依赖（feature gate）：默认使用 Fallback 分割器，启用 `neural` feature 后下载模型启用深度学习分割
