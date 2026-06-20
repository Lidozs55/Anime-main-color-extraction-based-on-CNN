"""
结果图生成。

提供两个独立函数,通常在 `python -m graphcolor.pipeline` 处理完图片后
单独调用以生成报告插图:

  - save_segmentation_visualization()
      保存主体识别可视化图,同时把"首次 mask"和"精修后 mask"叠在同一张图上,
      方便人工核对 refine 效果。

  - save_result_preview()
      在原图左下角叠加两个方形色块:前景主色、背景主色,
      输出与 student/preview.py 完全一致的格式。
"""
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np

from .preprocess import load_image
from .scoring import ImageResult
from .segment import SegmentResult


def save_segmentation_visualization(image_path: Union[str, Path],
                                    seg_result: SegmentResult,
                                    output_path: Union[str, Path]) -> str:
    """
    保存主体识别可视化图。

    - 首次提取mask：绿色半透明蒙版 + 绿色轮廓线
    - 精修后mask：蓝色半透明蒙版 + 红色轮廓线

    Args:
        image_path: 原始图片路径
        seg_result: 分割结果（含 initial_mask）
        output_path: 输出路径

    Returns:
        输出文件路径
    """
    bgr, _ = load_image(str(image_path))
    overlay = bgr.copy()

    refined_mask = seg_result.foreground_mask
    initial_mask = getattr(seg_result, 'initial_mask', None)

    # 缩放到原图尺寸
    if refined_mask.shape[:2] != bgr.shape[:2]:
        h, w = bgr.shape[:2]
        refined_mask = cv2.resize(refined_mask.astype(np.uint8), (w, h),
                                   interpolation=cv2.INTER_NEAREST).astype(bool)
    if initial_mask is not None and initial_mask.shape[:2] != bgr.shape[:2]:
        h, w = bgr.shape[:2]
        initial_mask = cv2.resize(initial_mask.astype(np.uint8), (w, h),
                                   interpolation=cv2.INTER_NEAREST).astype(bool)

    # 1. 首次提取mask：绿色蒙版
    if initial_mask is not None:
        overlay[initial_mask] = overlay[initial_mask] * 0.6 + np.array([0, 160, 0], dtype=np.uint8) * 0.4
        mask_u8 = initial_mask.astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 200, 0), 2)

    # 2. 精修后mask：蓝色蒙版（覆盖在绿色之上）
    if initial_mask is not None:
        # 只在精修后的区域内绘制蓝色（如果精修后比首次小，蓝色会显示"被裁切"的效果）
        refined_only = refined_mask & ~initial_mask  # 精修新增区域
        refined_common = refined_mask & initial_mask  # 两者重叠区域
        overlay[refined_common] = overlay[refined_common] * 0.5 + np.array([180, 80, 0], dtype=np.uint8) * 0.5
        if refined_only.any():
            overlay[refined_only] = overlay[refined_only] * 0.5 + np.array([180, 80, 0], dtype=np.uint8) * 0.5
    else:
        overlay[refined_mask] = overlay[refined_mask] * 0.5 + np.array([0, 180, 0], dtype=np.uint8) * 0.5

    # 绘制精修后的轮廓（红色）
    mask_u8 = refined_mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 0, 255), 2)

    # 添加文字说明
    method = seg_result.method
    ratio = seg_result.foreground_ratio
    if initial_mask is not None:
        initial_ratio = float(initial_mask.mean())
        label = f"Initial: {initial_ratio:.1%} (green) | Refined: {ratio:.1%} (blue) [{method}]"
    else:
        label = f"Foreground (method: {method}, ratio: {ratio:.1%})"
    
    cv2.putText(overlay, label, (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(overlay, label, (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ext = output_path.suffix.lower()
    if ext not in (".png", ".jpg", ".jpeg", ".bmp"):
        ext = ".png"
    ok = cv2.imwrite(str(output_path), overlay)
    if not ok:
        raise RuntimeError(f"无法保存可视化图: {output_path}")
    return str(output_path)


def save_result_preview(image_path: Union[str, Path],
                        result: ImageResult,
                        output_path: Union[str, Path],
                        swatch_size: Optional[int] = None) -> str:
    """
    在原始尺寸图片左下角叠加两个方形色块：前景主色、背景主色。
    """
    bgr, _ = load_image(str(image_path))
    h, w = bgr.shape[:2]
    canvas = bgr.copy()

    size = swatch_size or max(28, min(96, int(round(min(h, w) * 0.105))))
    gap = max(4, int(round(size * 0.16)))
    margin = max(8, int(round(size * 0.22)))
    stroke = max(2, int(round(size * 0.045)))

    colors = [
        result.foreground.dominant_color.rgb,
        result.background.dominant_color.rgb,
    ]

    y1 = max(0, h - margin - size)
    y2 = min(h, y1 + size)
    for idx, rgb in enumerate(colors):
        x1 = margin + idx * (size + gap)
        x2 = min(w, x1 + size)
        if x1 >= w:
            break

        bgr_color = tuple(int(v) for v in rgb[::-1])
        cv2.rectangle(canvas, (x1, y1), (x2 - 1, y2 - 1), bgr_color, cv2.FILLED)

        # Dual stroke stays visible on both dark and light artwork.
        cv2.rectangle(canvas, (x1, y1), (x2 - 1, y2 - 1), (255, 255, 255), stroke)
        cv2.rectangle(canvas, (x1, y1), (x2 - 1, y2 - 1), (0, 0, 0), max(1, stroke // 2))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ext = output_path.suffix.lower()
    if ext not in (".png", ".jpg", ".jpeg", ".bmp"):
        ext = ".png"
    ok = cv2.imwrite(str(output_path), canvas)
    if not ok:
        raise RuntimeError(f"无法保存结果图: {output_path}")
    return str(output_path)
