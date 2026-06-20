"""
student - 学生模型(CNN)蒸馏模块。

把 graphcolor 教师模型的"前景/背景主色"能力蒸馏到一个 ~3MB 的轻量 CNN
中,使得在普通机器甚至无 Python 环境下也能毫秒级预测主色。

模块组成:
  model.py    —— ColorNet-Masked 模型定义(MBConv 主干 + 软掩膜 + 双 FC 头)
  dataset.py  —— ColorDataset 数据集类(支持本地图片 + Pixiv 图片 + targets JSON)
  train.py    —— 训练入口(AdamW + CosineAnnealing + early stopping + 断点续训)
  eval.py     —— 评估入口(在 val 集上计算 dE*ab)
  export.py   —— 导出 TorchScript 模型 (colornet_masked.pt)
  preview.py  —— 批量出图,左下角叠加 fg / bg 两个主色色块(与 graphcolor 风格一致)

数据契约:
  - 训练数据来源: 根目录的 img/ + pixiv_img/
  - 标签来源:     根目录的 targets.json / targets_*.json / targets_pixiv_*.json
  - 监督信号:     targets 中每条记录的 L_fg/a_fg/b_fg / L_bg/a_bg/b_bg
"""
