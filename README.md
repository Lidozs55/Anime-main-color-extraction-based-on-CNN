# GraphColor + CNN

主色提取（教师模型 `graphcolor/`）+ CNN 蒸馏（学生模型 `student/`）+ Web 标注工具（`label.py`）的合并项目。

## 项目简介

本项目合并了两个子项目：
- **Graph-Color-Labeling-APP**：主色提取 + Web 标注 App
- **Anime-main-color-extraction-based-on-CNN**：CNN 学生模型蒸馏

合并后，根目录下提供了统一的项目结构与单文件标注入口 `label.py`，同时保留了 CNN 训练与预览的全部功能。

## 特性

- **多模式标注工具**（`label.py`）：Pixiv 流式 / Pixiv 持久化 / 本地图片 / 本地 results.json 四种入口。
- **断点续标**：自动保存至 `label_progress.json`，重启后继续；输出文件 `targets_{TIME}.json` 按时间戳区分会话。
- **教师模型**（`graphcolor/`）：基于图论的主色提取（pipeline + cluster + scoring + 可选 NeuralSegmenter）。
- **学生模型**（`student/`）：ColorNet-Masked，蒸馏主色预测，支持 `train.py` / `eval.py` / `export.py` / `preview.py`。
- **Web 标注界面**：深色主题 + 候选色块 + 调色盘 + hex 输入，跳过/退出按钮齐全。
- **单 exe 打包**：通过 `label_APP.spec` 使用 PyInstaller 打包独立 exe，无需 Python 环境即可标注 Pixiv 图源（默认模式）。

## 安装

```bash
# 必需依赖
pip install -r requirements.txt
```

`torch`/`torchvision` 仅在训练 / 推理学生模型时必需；`rembg`/`onnxruntime` 仅在使用 `NeuralSegmenter` 时需要。

## 快速开始

### 1. Pixiv 流式标注（默认模式）

```bash
python label.py
```

- 启动后浏览器自动打开 `http://localhost:5000`。
- 程序通过 lolicon API 随机拉取 Pixiv 图片（默认过滤 R-18），下载到 `pixiv_temp/`，由 graphcolor 提取主色后入队。
- 教师置信度高时自动通过；低时弹出 Web 端让你选择/跳过。
- 每标注完一张图，临时图片立即删除；会话退出时整个 `pixiv_temp/` 会被清空。
- 进度自动写入 `targets_{TIME}.json` 与 `label_progress.json`。

### 2. Pixiv 持久化标注（`--quick` 模式）

```bash
python label.py --quick
```

- 图片直接下载到 `pixiv_img/`（不会被删除），适合一边标注一边积累训练数据。
- **跳过时会从 `pixiv_img/` 中删除对应文件**（避免把"被跳过的废图"喂给 CNN）。
- 标注完成的图片保留在 `pixiv_img/`，可直接用于学生模型训练。

### 3. 本地图片标注

```bash
python label.py img/*.png
# 或
python label.py path/to/dataset_dir/
# 或
python label.py data.zip
```

- 支持通配符、目录、zip 压缩包。
- 复用 `graphcolor.process()` 提取主色，触发人工标注阈值与 Pixiv 模式相同。
- 输出 `targets_{TIME}.json`（key 为文件 basename）。
- 不修改原图（用户输入的目录与文件保持原样）。

### 4. 本地 results 标注

```bash
python label.py outputs/results.json
```

- 直接读取已有的 results.json（list of dict，每条含 `foreground.main_colors` 与 `background.main_colors`），跳过 graphcolor 计算。
- 仅做人工标注环节，输出到 `targets_{TIME}.json`。

### 5. 迁移旧版置信度

```bash
python label.py --migrate targets.json
```

- 将旧版 `fg_conf` / `bg_conf` 数值（< 0.52 表示人工标注）转换为新版本公式 `min(score1 / (2 * score2), 1.0)`。

## 学生模型训练与预览

### 训练

```bash
cd student
python train.py --episodes 60 --patience 15
```

- 默认从根目录的 `img/` 读取本地图片，从 `pixiv_img/` 读取 Pixiv 图片。
- 自动发现 `targets.json` 与 `targets_*.json` / `targets_pixiv_*.json`。
- 训练产物：`student/checkpoint.pth`、`student/best_model.pth`。

### 评估

```bash
cd student
python eval.py
```

### 导出 ONNX

```bash
cd student
python export.py
```

### 批量预览

```bash
cd student
python preview.py
```

- 默认从 `../img` 与 `../pixiv_img` 收集所有 jpg/png/jpeg/webp/bmp。
- 输出至 `../outputs/model_previews/`。

## 目录结构

```
.
├── graphcolor/                  # 教师模型：主色提取算法
│   ├── __init__.py
│   ├── pipeline.py
│   ├── preprocess.py
│   ├── segment.py
│   ├── cluster.py
│   ├── scoring.py
│   ├── visualize.py
│   └── html_visualize.py
├── student/                     # 学生模型：CNN 蒸馏
│   ├── __init__.py
│   ├── model.py
│   ├── dataset.py
│   ├── train.py
│   ├── eval.py
│   ├── export.py
│   └── preview.py
├── templates/
│   └── index.html               # Web 标注前端
├── img/                         # 本地图片（无子文件夹，供 label.py 本地模式 + student/preview.py）
├── pixiv_img/                   # Pixiv 持久化图片（CNN 训练 + --quick 模式）
├── pixiv_temp/                  # Pixiv 临时图片（label.py 默认模式，结束清空）
├── label.py                     # 标注工具（前后端一体）
├── label_APP.spec               # PyInstaller 单 exe 打包配置
├── requirements.txt
├── README.md
└── .gitignore
```

## 断点 / 续训

每次标注完成一张图，`label.py` 会：
1. 增量写一次 `targets_{TIME}.json`（本次会话唯一输出）。
2. 完整写一次 `label_progress.json`（含 session_id、download pool、annotation queue、stats）。

下次启动 `label.py`：
- 若 `session_id` 与当前会话相同 → 自动恢复 targets / pool / queue 继续。
- 若 `session_id` 不同 → 提示"上次会话于 X 已保存 N 条"并备份为 `label_progress_{OLD_SESSION}.json.bak`。
- 若 `label_progress.json` 不存在 → 全新会话，生成新的 `targets_{TIME}.json`。

按 Ctrl+C 或点击 Web 端"退出系统"按钮都会触发 `save_full_checkpoint()` 后再退出。

## 打包单 exe

```bash
# 准备虚拟环境（可选）
python -m venv build_env
.\build_env\Scripts\activate
pip install -r requirements.txt

# 打包
pyinstaller label_APP.spec --noconfirm
```

打包后输出 `dist/label_APP/label_APP.exe`（包含 templates/）。默认 console=False，**双击后自动打开浏览器**开始 Pixiv 标注（默认模式，不带 `--quick`）。

要分发给无 Python 环境的人，把 `dist/label_APP/` 整个目录打包即可。

## 算法概要

### graphcolor（教师模型）

1. **preprocess**：缩放 + 转换到 Lab 色彩空间。
2. **segment**：基于 `rembg` 神经网络或 GrabCut 提取前景蒙版。
3. **cluster**：在 Lab 空间使用 Mini-Batch K-Means 聚类（`scikit-learn`）。
4. **scoring**：综合 chroma + lightness 双调权打分（`color_weight=1.0`, `lightness_weight=0.5`），可选 bilinear 权重与 skin 惩罚。
5. **visualize**：保存标注结果预览图与色块可视化图。

置信度公式：`min(score1 / (2 * score2), 1.0)`。当 `gap12 < 0.08` 或 `gap13 < 0.15` 时置信度过低，触发人工标注。

### student（学生模型）

`ColorNet-Masked`（轻量 CNN，~3MB），输入 `128×128` RGB + 前景 mask（1ch），输出 3 维（Lab 空间的 L/a/b）。训练采用 MSE Loss + L1 蒸馏自教师预测，支持数据增强（随机水平翻转、颜色抖动）。

训练数据：根目录 `img/` 下的本地图片 + `pixiv_img/` 下的 Pixiv 图片，按 `targets_*.json` 中的 `L_fg/a_fg/b_fg` 与 `L_bg/a_bg/b_bg` 监督。

## 关键差异

| 旧子项目 | 合并后 |
|---------|--------|
| `Graph-Color-Labeling-APP/main.py` (Pixiv 弹窗) | 整合至 `label.py`（Web 标注） |
| `Anime-main-color-extraction-based-on-CNN/generate_targets.py` (本地弹窗) | 整合至 `label.py`（Web 标注） |
| `Anime-main-color-extraction-based-on-CNN/generate_targets_pixiv.py` (Pixiv 弹窗) | 整合至 `label.py`（Web 标注） |
| `extracted_imgs/imgs/pixiv_imgs/` | `pixiv_img/`（独立目录，无子目录） |
| `extracted_imgs/imgs/` (本地) | `img/`（独立目录，无子目录） |
| `targets_pixiv_progress.json` | `label_progress.json`（统一断点） |
| 旧子目录 `Graph-Color-Labeling-APP/` `Anime-main-color-extraction-based-on-CNN/` | 保留供对照与回滚 |

## License

本仓库基于 MIT License 发布。
