"""
graphcolor - 教师模型（基于图论/Lab 空间的主色提取算法包）

本包实现了从一张图片中提取"前景主色 + 背景主色"的完整管线:

  pipeline   —— 入口编排,串联以下四个模块对单张/多张图片进行处理
  preprocess —— 加载图片、等比缩放、BGR↔Lab 色彩空间转换
  segment    —— 主体/背景分离(提供基于 rembg 的 NeuralSegmenter 和
                基于 OpenCV GrabCut 的 ForegroundSegmenter 两种实现,
                还包含 GrabCut 空洞精修逻辑)
  cluster    —— 在加权 Lab 空间下做 Mini-Batch K-Means 聚类
                (颜色优先、亮度辅助,适当放大 a* 红绿轴)
  scoring    —— 综合"像素占比 + 亮度均匀性 + 空间中心距离 + 视觉显著性"
                的多因子评分,并支持肤色折扣;还提供 analyze_region 便捷
                函数与 MainColor / RegionResult / ImageResult 数据类
  visualize       —— 将分割结果 / 主色色块保存为 PNG
  html_visualize  —— 把一批结果流式生成静态 HTML 报告
  shadow          —— 纯经典 Lab 空间阴影去除(无神经网络,主色提取前调用)

典型用法:
    from graphcolor import GraphColorPipeline
    pipe = GraphColorPipeline()
    pipe.warmup()
    result = pipe.process("image.png")
    print(result.foreground.dominant_color.hex_color)
"""
from .pipeline import GraphColorPipeline, process_batch
from .preprocess import load_and_resize
from .segment import ForegroundSegmenter, NeuralSegmenter
from .cluster import LabClusterer
from .scoring import ClusterScorer
from .shadow import ShadowRemover
from .visualize import save_result_preview, save_segmentation_visualization

__version__ = "0.1.0"
