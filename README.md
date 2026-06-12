# GraphColor

手绘动漫图片主色提取工具。从动漫/插画中智能提取前景（主体）和背景的主色调，输出颜色评分、色值及可视化结果。

包含一个**学生模型蒸馏管线**：通过教师模型（graphcolor pipeline）生成软目标，训练轻量 CNN 直接从图片预测前景/背景主色。

## 特性

- **智能主体分割**：支持多种分割方案，按优先级回退
  - Alpha 通道优先（PNG 透明图直接提取）
  - 深度学习分割：基于 rembg + U2-Net/IS-Net，对动漫/插画等复杂背景场景识别更精准
  - 传统多策略融合（Fallback）：色度检测 + 边缘检测 + 亮度分析 + GrabCut 精修
- **空洞检测与精修**：自动检测环状主体（如张开的手臂、头发遮挡）内部的背景穿透区域，使用 GrabCut 精修 mask
- **Lab 空间加权聚类**：对 Lab 三通道进行非对称加权（L\*=0.5, a\*=1.1, b\*=1.0），强化色度差异、弱化亮度影响，使相近色相不同明度的颜色更容易归为一类
- **多维度评分体系**：综合像素占比、亮度均匀性、空间中心距离、视觉显著性四个因子，对每个聚类进行综合评分
- **批量处理**：支持多图片/通配符/zip 批量输入，支持多进程并行
- **可视化输出**：生成带主色色块的结果图与主体蒙版分割可视化图
- **学生模型蒸馏**：轻量 CNN（~43K 参数）通过 confidence-weighted loss 从教师模型学习，支持梯度累积全量训练

## 安装

### 依赖

```
numpy>=1.24.0
opencv-python-headless>=4.8.0
scikit-learn>=1.3.0
Pillow>=10.0.0
```

### 可选依赖（深度学习分割 + 学生模型训练）

```
pip install rembg torch torchvision
```

### 安装步骤

```bash
pip install -r requirements.txt
# 可选：安装 rembg（深度学习分割）和 PyTorch（学生模型训练）
pip install rembg torch torchvision
```

## 快速开始

### 处理单张图片

```bash
python main.py image.png
```

### 批量处理

```bash
python main.py img1.jpg img2.jpg --output results.json
```

### 使用通配符

```bash
python main.py "images/*.png" --output batch_result.json
```

### 处理 zip 中的图片

```bash
python main.py images.zip --extract-dir ./extracted_imgs
```

### 带可视化输出

```bash
python main.py image.png --visual-dir ./previews --seg-visual-dir ./seg_visuals
```

### 并行处理

```bash
python main.py "images/*.png" -j 4 --output results.json
```

## 学生模型蒸馏管线

### 整体流程

```
教师模型 (graphcolor pipeline)
  │
  ▼
generate_targets.py / generate_targets_pixiv.py
  │  从 results.json 中选取人类认可的颜色作为 target
  │  每个 target 包含: L/a/b × 前景/背景 + 置信度 (fg_conf/bg_conf)
  │
  ▼
targets.json / targets_pixiv_*.json
  │
  ▼
student/train.py
  │  梯度累积全量训练 + confidence-weighted loss
  │  val_loss 模型选择 + early stopping
  │
  ▼
student/best_model.pth
  │
  ▼
student/eval.py  →  单图推理验证
student/export.py → 导出模型信息
```

### 生成蒸馏目标

**本地图片**（从 `outputs/results.json` 交互校正）：

```bash
# 先运行教师模型
python main.py "images/*.png" -o outputs/results.json

# 交互校正生成 targets.json
python generate_targets.py
```

**Pixiv 图片**（自动下载 + 交互校正）：

```bash
# 循环下载 Pixiv 随机图片，每批 20 张
python generate_targets_pixiv.py

# Quick 模式：保留图片到 extracted_imgs/imgs/pixiv_imgs/
python generate_targets_pixiv.py --quick
```

**人工标注触发条件**：
- `gap12 < 0.05`：前 2 名分数差距不足 5%，难以自动区分
- `gap13 < 0.20`：前 3 名分数差距不足 20%，候选颜色区分度不足

可选颜色上限为 5 个（pipeline 输出最多 5 个主色），不足 5 个时显示实际数量。

### 训练学生模型

```bash
cd student

# 从头训练
python train.py --episodes 200 --patience 30

# 断点续训（自动从最佳模型恢复）
python train.py --episodes 200 --patience 50

# 自定义参数
python train.py --lr 5e-4 --batch-size 16 --ema-alpha 0.1
```

**训练策略**：

| 特性 | 说明 |
|------|------|
| 梯度累积 | 遍历全集后统一 update，消除 batch 分布偏差 |
| Confidence-weighted loss | 高置信样本梯度更大，低置信样本被降权 |
| 检查点恢复 | 从最佳模型权重重新开始，重置 optimizer/scheduler |
| 模型选择 | 基于 val_loss（70/30 拆分的真实泛化信号） |
| 正则化 | Dropout(0.2) + WeightDecay(1e-4) + 梯度裁剪(5.0) |

**命令行参数**：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--episodes` | 200 | 训练总 epoch 数 |
| `--batch-size` | 32 | mini-batch size（梯度累积用） |
| `--patience` | 30 | early stopping patience |
| `--lr` | 1e-3 | 学习率 |
| `--ema-alpha` | 0.1 | EMA 平滑系数 |

### 评估与导出

```bash
cd student

# 单图评估
python eval.py --image ../path/to/image.png

# 导出模型信息
python export.py
```

## 学生模型架构

基于 MobileNetV2 风格的 MBConv（Mobile Bottleneck Convolution）：

```
Input (3×128×128)
  │
  ▼
Stem: Conv2d(3→16, k3, s2) + BN + ReLU6
  │
  ▼
MBConv Block1: 16→16, s1, expand=2  ─┐
MBConv Block2: 16→24, s2, expand=4   │ 提取特征
MBConv Block3: 24→24, s1, expand=4   │ (逐层升维)
MBConv Block4: 24→32, s2, expand=6   │
MBConv Block5: 32→32, s1, expand=6  ─┘
  │
  ▼
Fusion: Conv2d(32→64, k1) + BN + ReLU6
  │
  ├─→ Mask Head: Conv2d(64→1, k1) + Sigmoid → 前景/背景权重图
  │
  ├─→ FG FC: GAP(mask×feat) → Linear(64→32) → Dropout(0.2) → Linear(32→3) → 前景 Lab
  │
  └─→ BG FC: GAP((1-mask)×feat) → Linear(64→32) → Dropout(0.2) → Linear(32→3) → 背景 Lab
```

- 总参数量：~43K
- 输出：前景 Lab (L/a/b) + 背景 Lab (L/a/b) + 分割 mask
- L 通道范围 [0, 100]（sigmoid × 100），a/b 通道范围 [-128, 128]（tanh × 128）

## 命令行参数

| 参数 | 简写 | 默认值 | 说明 |
|------|------|--------|------|
| `images` | - | 必填 | 图片路径，支持通配符和目录 |
| `--output` | `-o` | - | 输出 JSON 文件路径 |
| `--visual-dir` | - | - | 结果图输出目录（左下角叠加主色色块） |
| `--seg-visual-dir` | - | - | 主体识别可视化输出目录 |
| `--extract-dir` | - | - | zip 解压目录 |
| `--max-size` | - | 512 | 缩放后图片最长边像素数 |
| `--clusters-fg` | - | 10 | 前景聚类数 |
| `--clusters-bg` | - | 6 | 背景聚类数 |
| `--color-weight` | - | 1.0 | 颜色通道（a\*/b\*）基权重 |
| `--lightness-weight` | - | 0.5 | 亮度通道（L\*）权重 |
| `--a-boost` | - | 1.1 | a\* 通道额外放大倍数 |
| `--workers` | `-j` | 1 | 并行进程数 |

## 处理流程

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
[3] BGR → Lab 转换：将颜色空间转换到 CIE Lab（感知均匀）
  │
  ▼
[4] 主体提取（按优先级回退）
  │   1. Alpha 通道分割（最可靠）
  │   2. 深度学习显著性分割（rembg）
  │   3. 多策略融合分割（Fallback：色度+边缘+亮度+GrabCut）
  │
  ▼
[5] 背景主色计算
  │   背景像素 → 聚类(k=6) → 四因子评分 → 取 Top N 背景主色
  │
  ▼
[6] 空洞检测 + GrabCut 精修
  │   前景像素对比背景主色 → 找空洞 → 连通域过滤 → GrabCut 精修 mask
  │
  ▼
[7] 前景主色计算
  │   精修后前景像素 → 聚类(k=10) → 四因子评分 → 输出主色
  │
  ▼
[8] 输出 ImageResult（JSON + 终端 + 可视化图）
```

## 评分体系

每个聚类的主色评分由以下四个因子加权计算：

| 因子 | 权重 | 说明 |
|------|------|------|
| 像素占比 | ~0.476 | 该颜色在区域中的覆盖率 |
| 亮度均匀性 | ~0.095 | 亮度方差越小（颜色越纯净）分越高 |
| 空间中心距离 | ~0.095 | 越靠近图像中心权重越高 |
| 视觉显著性 | ~0.286 | 融合加权色度和极端亮度，鲜艳色 > 极端明暗 > 中间灰 |

**视觉显著性**是最复杂的因子，通过 sigmoid 权重分配色度与亮度的贡献（默认色度权重约 88%，亮度权重约 12%），确保鲜艳颜色优先被识别为主色。

## 输出格式

### JSON 输出示例

```json
[
  {
    "image": "sample.png",
    "segment_method": "neural_isnet-general-use",
    "foreground": {
      "dominant_color": {
        "lab": [45.2, 18.5, -22.3],
        "rgb": [142, 89, 156],
        "hex": "#8E599C",
        "score": 0.6234,
        "proportion": 0.3812
      },
      "main_colors": [...]
    },
    "background": {
      "dominant_color": {...},
      "main_colors": [...]
    }
  }
]
```

### targets.json 示例

```json
{
  "image.png": {
    "L_fg": 45.2, "a_fg": 18.5, "b_fg": -22.3, "fg_conf": 0.7234,
    "L_bg": 85.0, "a_bg": -2.1, "b_bg": 15.6, "bg_conf": 0.6512
  }
}
```

## 项目结构

```
GraphColor/
├── graphcolor/                 # 教师模型（主色提取 pipeline）
│   ├── __init__.py
│   ├── pipeline.py             # 处理管线编排（主流程）
│   ├── preprocess.py           # 加载、缩放、Lab 转换
│   ├── segment.py              # 主体/背景分割
│   ├── cluster.py              # Lab 空间加权聚类
│   ├── scoring.py              # 四因子评分与主色提取
│   ├── visualize.py            # 结果可视化
│   └── html_visualize.py       # HTML 可视化
│
├── student/                    # 学生模型（CNN 蒸馏）
│   ├── model.py                # ColorNetMasked 架构（MBConv + mask head）
│   ├── dataset.py              # ColorDataset（支持 Pixiv URL 自动下载）
│   ├── train.py                # 训练脚本（梯度累积 + confidence-weighted loss）
│   ├── eval.py                 # 单图评估脚本
│   └── export.py               # 模型导出脚本
│
├── main.py                     # 教师模型 CLI 入口
├── generate_targets.py         # 本地图片蒸馏目标生成（交互校正）
├── generate_targets_pixiv.py   # Pixiv 图片蒸馏目标生成（自动下载 + 交互校正）
├── requirements.txt
├── GRAPHCOLOR_SPEC.md          # 完整算法规范
└── README.md
```

## 算法规范

详细的算法原理、关键公式和复现指南请见 [GRAPHCOLOR_SPEC.md](GRAPHCOLOR_SPEC.md)，包含：

- 完整的色彩空间转换公式（sRGB → XYZ → Lab）
- Lab 加权聚类距离公式
- 四因子评分的详细数学推导
- 视觉显著性融合公式
- 空洞检测与 GrabCut 精修原理
- 所有配置参数的默认值与调优建议

## 作为库使用

```python
from graphcolor import GraphColorPipeline

pipeline = GraphColorPipeline()
result = pipeline.process("image.png")

print(f"前景主色: {result.foreground.dominant_color.hex_color}")
print(f"  Lab: {result.foreground.dominant_color.lab}")
print(f"  评分: {result.foreground.dominant_color.score}")
print(f"  占比: {result.foreground.dominant_color.proportion}")
```

批量处理：

```python
from graphcolor.pipeline import process_batch

results = process_batch(
    ["img1.png", "img2.jpg"],
    config={"max_size": 256, "n_clusters_foreground": 8},
    output_json="results.json",
    output_preview_dir="./previews",
    verbose=True,
    workers=4
)
```

## 配置调优指南

### 分割参数（控制主体识别的激进程度）

| 参数 | 调低效果 | 调高效果 |
|------|----------|----------|
| `chroma_threshold` | 更多彩色区域被纳入主体 | 主体识别更保守 |
| `min_contour_area_ratio` | 保留更小的轮廓细节 | 只保留大块主体 |
| `foreground_score_threshold` | 主体识别更激进 | 主体识别更严格 |

### 聚类参数

| 参数 | 说明 | 建议 |
|------|------|------|
| `color_weight` | 颜色通道权重 | 增大 → 聚类更关注色相 |
| `lightness_weight` | 亮度权重 | 降低（0.1~0.3）使相近色相不同明度归为一类 |
| `a_boost` | a\* 通道放大 | 略大于 1.0 可提升红绿区分度 |

### 评分参数

| 参数 | 说明 | 建议 |
|------|------|------|
| `weight_count` | 像素占比权重 | 增大 → 大面积颜色优先 |
| `weight_variance` | 亮度均匀性权重 | 增大 → 颜色纯净度优先 |
| `weight_center` | 空间中心权重 | 增大 → 靠近中心的颜色优先 |
| `weight_chroma` | 视觉显著性权重 | 增大 → 鲜艳颜色优先 |

### 空洞检测参数

| 参数 | 说明 | 建议 |
|------|------|------|
| `hole_distance_threshold` | BGR 欧氏距离阈值 | 提高（20~25）→ 更激进的空洞检测 |
| `grabcut_iterations` | GrabCut 迭代次数 | 1~3，一般 1 次足够 |

## License

MIT
