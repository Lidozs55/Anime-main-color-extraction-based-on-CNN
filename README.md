# GraphColor

手绘动漫图片主色提取工具。从动漫/插画中智能提取前景（主体）和背景的主色调，输出颜色评分、色值及可视化结果。

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

## 安装

### 依赖

```
numpy>=1.24.0
opencv-python-headless>=4.8.0
scikit-learn>=1.3.0
Pillow>=10.0.0
```

### 可选依赖（深度学习分割）

如需使用默认的深度学习分割器，还需安装：

```
pip install rembg
```

### 安装步骤

```bash
pip install -r requirements.txt
# 可选：安装 rembg（深度学习分割）
pip install rembg
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
| `--color-weight` | - | 1.0 | 颜色通道（a*/b*）基权重 |
| `--lightness-weight` | - | 0.5 | 亮度通道（L*）权重 |
| `--a-boost` | - | 1.1 | a* 通道额外放大倍数 |
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

## 项目结构

```
graphcolor/
├── __init__.py         # 模块入口
├── pipeline.py         # 处理管线编排（主流程）
├── preprocess.py       # 加载、缩放、Lab 转换
├── segment.py          # 主体/背景分割（NeuralSegmenter + ForegroundSegmenter）
├── cluster.py          # Lab 空间加权聚类
├── scoring.py          # 四因子评分与主色提取
└── visualize.py        # 结果可视化（色块叠加、分割蒙版可视化）
main.py                 # CLI 入口
requirements.txt        # Python 依赖
GRAPHCOLOR_SPEC.md      # 完整算法规范（含所有公式）
test_pipeline.py        # 测试脚本
analyze_fg.py           # 前景主色分析
analyze_bg.py           # 背景主色分析
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
| `a_boost` | a* 通道放大 | 略大于 1.0 可提升红绿区分度 |

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
