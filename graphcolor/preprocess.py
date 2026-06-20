"""
图片加载、压缩与色彩空间转换模块。

整条管线最底层的 IO / 色彩空间工具集:
  - load_image()       读取图片文件,处理 Windows 下的非 ASCII 路径问题,
                       对带 alpha 的 PNG 在白底上预合成(透明像素仍通过 alpha 单独返回)
  - resize_to_max()    等比缩放,默认用 INTER_AREA(下采样抗锯齿效果好)
  - load_and_resize()  上面两个的一步组合便捷函数
  - bgr_to_lab()       OpenCV Lab 像素格式(0~255) -> 标准 Lab(L:0~100, a/b:-128~127)
  - lab_to_bgr()       反向,标准 Lab -> BGR uint8
  - lab_to_rgb_for_display()  适用于"一组像素"(N,3) 的批量可视化

主色提取的所有算法都在 Lab 空间运行,所以这一步的色彩空间归一化至关重要。
"""
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple


def load_image(image_path: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    加载原图，返回 BGR 图像和可选 alpha 通道。

    使用 imdecode + fromfile，避免 Windows 下非 ASCII 路径被 cv2.imread 读坏。
    若 PNG 带透明通道，会先合成到白底，透明像素仍通过 alpha 返回给分割模块。
    """
    data = np.fromfile(str(image_path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"无法加载图片: {image_path}")

    alpha = None
    if img.ndim == 2:
        bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        bgr_raw = img[:, :, :3].astype(np.float32)
        alpha = img[:, :, 3]
        a = (alpha.astype(np.float32) / 255.0)[:, :, None]
        bgr = (bgr_raw * a + 255.0 * (1.0 - a)).astype(np.uint8)
    else:
        bgr = img[:, :, :3]

    return bgr, alpha


def resize_to_max(img: np.ndarray, max_size: int = 512,
                  interpolation: int = cv2.INTER_AREA) -> np.ndarray:
    """等比缩放到最长边不超过 max_size。"""
    h, w = img.shape[:2]
    if max(h, w) <= max_size:
        return img.copy()

    scale = max_size / max(h, w)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(img, (new_w, new_h), interpolation=interpolation)


def load_and_resize(image_path: str, max_size: int = 512) -> np.ndarray:
    """
    加载图片并等比缩放到 max_size 以内。

    Args:
        image_path: 图片路径
        max_size: 最长边的最大像素数

    Returns:
        BGR格式的numpy数组 (H, W, 3)
    """
    img, _ = load_image(image_path)
    return resize_to_max(img, max_size)


def bgr_to_lab(bgr_img: np.ndarray) -> np.ndarray:
    """
    BGR -> CIE Lab (float32, L范围0~100, a/b范围~[-128,127])
    """
    lab = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2Lab).astype(np.float32)
    # OpenCV Lab: L 0-255, a 0-255, b 0-255
    # 映射到标准 Lab: L 0-100, a -128~127, b -128~127
    lab[:, :, 0] = lab[:, :, 0] / 255.0 * 100.0      # L: 0 -> 100
    lab[:, :, 1] = lab[:, :, 1] - 128.0               # a: -128 -> 127
    lab[:, :, 2] = lab[:, :, 2] - 128.0               # b: -128 -> 127
    return lab


def lab_to_bgr(lab_img: np.ndarray) -> np.ndarray:
    """
    标准 Lab -> BGR (uint8)
    """
    lab = lab_img.copy()
    lab[:, :, 0] = lab[:, :, 0] / 100.0 * 255.0
    lab[:, :, 1] = lab[:, :, 1] + 128.0
    lab[:, :, 2] = lab[:, :, 2] + 128.0
    lab = np.clip(lab, 0, 255).astype(np.uint8)
    return cv2.cvtColor(lab, cv2.COLOR_Lab2BGR)


def lab_to_rgb_for_display(lab_pixels: np.ndarray) -> np.ndarray:
    """Lab像素点 (N,3) -> RGB uint8，用于可视化"""
    lab_2d = lab_pixels.reshape(-1, 1, 3).astype(np.float32)
    lab_2d[:, :, 0] = lab_2d[:, :, 0] / 100.0 * 255.0
    lab_2d[:, :, 1] = lab_2d[:, :, 1] + 128.0
    lab_2d[:, :, 2] = lab_2d[:, :, 2] + 128.0
    lab_2d = np.clip(lab_2d, 0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(lab_2d, cv2.COLOR_Lab2BGR)
    return bgr[:, 0, ::-1]  # BGR -> RGB
