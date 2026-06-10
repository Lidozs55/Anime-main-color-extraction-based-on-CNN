#!/usr/bin/env python
"""
分析单张图片的背景像素分布，与JSON结果中的背景主色进行对比。
"""
import json
import numpy as np
import cv2
from pathlib import Path

from graphcolor.preprocess import load_image, bgr_to_lab, lab_to_rgb_for_display
from graphcolor.segment import NeuralSegmenter


def analyze_image_background(image_path: str, json_path: str = "outputs/results.json"):
    # 加载JSON结果
    with open(json_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    # 查找目标图片的结果
    target = None
    for r in results:
        if "102548643" in r["image"]:
            target = r
            break

    if target is None:
        print("未找到图片 102488806 的结果")
        return

    print("=" * 70)
    print(f"分析图片: {target['image']}")
    print(f"分割方法: {target['segment_method']}")
    print("=" * 70)

    # 打印JSON中的背景主色
    bg_colors = target["background"]["main_colors"]
    print(f"\nJSON中的背景主色 ({len(bg_colors)}个):")
    for i, c in enumerate(bg_colors):
        print(f"  [{i}] {c['hex']} | RGB {c['rgb']} | Lab {c['lab']} | "
              f"score={c['score']:.4f} | proportion={c['proportion']:.4f}")

    # 加载原图
    bgr, alpha = load_image(image_path)
    lab = bgr_to_lab(bgr)
    h, w = bgr.shape[:2]
    print(f"\n图片尺寸: {w}x{h}")

    # 分割
    segmenter = NeuralSegmenter()
    seg_result = segmenter.segment(lab, bgr_img=bgr, alpha_mask=alpha)
    bg_mask = seg_result.background_mask
    fg_ratio = seg_result.foreground_ratio
    bg_pixels = lab[bg_mask]
    print(f"前景占比: {fg_ratio:.1%} | 背景占比: {1-fg_ratio:.1%}")
    print(f"背景像素数: {len(bg_pixels)}")

    # 背景像素的RGB均值和中位数
    bg_rgb = lab_to_rgb_for_display(bg_pixels)
    print(f"\n背景像素 RGB 统计:")
    print(f"  均值:   R={bg_rgb[:, 0].mean():.1f}, G={bg_rgb[:, 1].mean():.1f}, B={bg_rgb[:, 2].mean():.1f}")
    print(f"  中位数: R={np.median(bg_rgb[:, 0]):.1f}, G={np.median(bg_rgb[:, 1]):.1f}, B={np.median(bg_rgb[:, 2]):.1f}")
    print(f"  标准差: R={bg_rgb[:, 0].std():.1f}, G={bg_rgb[:, 1].std():.1f}, B={bg_rgb[:, 2].std():.1f}")

    # 背景像素的Lab统计
    print(f"\n背景像素 Lab 统计:")
    print(f"  均值:   L={bg_pixels[:, 0].mean():.1f}, a={bg_pixels[:, 1].mean():.1f}, b={bg_pixels[:, 2].mean():.1f}")
    print(f"  中位数: L={np.median(bg_pixels[:, 0]):.1f}, a={np.median(bg_pixels[:, 1]):.1f}, b={np.median(bg_pixels[:, 2]):.1f}")
    print(f"  标准差: L={bg_pixels[:, 0].std():.1f}, a={bg_pixels[:, 1].std():.1f}, b={bg_pixels[:, 2].std():.1f}")

    # 分析各主色在背景中的匹配情况（以聚类中心为基准，计算匹配像素比例）
    print(f"\n各主色匹配分析 (基于Lab空间距离):")
    bg_centers_lab = np.array([c["lab"] for c in bg_colors])

    # 计算每个背景像素到各主色的Lab距离
    from scipy.spatial.distance import cdist
    distances = cdist(bg_pixels, bg_centers_lab, metric='euclidean')

    # 每个像素归属最近的主色
    nearest = np.argmin(distances, axis=1)

    # 设置一个匹配阈值（Lab距离 < 25 视为匹配该主色）
    threshold = 25.0
    for i, c in enumerate(bg_colors):
        mask = nearest == i
        count = mask.sum()
        proportion = count / len(bg_pixels)
        matched_mask = distances[:, i] < threshold
        matched_count = matched_mask.sum()
        matched_proportion = matched_count / len(bg_pixels)

        # 该主色匹配区域的像素统计
        if matched_count > 0:
            matched_pixels = bg_pixels[matched_mask]
            matched_rgb = bg_rgb[matched_mask]
            print(f"\n  [{i}] {c['hex']} (JSON proportion={c['proportion']:.4f}):")
            print(f"      最近归属: {count} 像素 ({proportion:.4f})")
            print(f"      Lab<25匹配: {matched_count} 像素 ({matched_proportion:.4f})")
            print(f"      匹配区域 RGB 均值: R={matched_rgb[:, 0].mean():.1f}, G={matched_rgb[:, 1].mean():.1f}, B={matched_rgb[:, 2].mean():.1f}")
            print(f"      匹配区域 RGB 中位数: R={np.median(matched_rgb[:, 0]):.0f}, G={np.median(matched_rgb[:, 1]):.0f}, B={np.median(matched_rgb[:, 2]):.0f}")
        else:
            print(f"\n  [{i}] {c['hex']}: 无匹配像素")

    # 整体匹配度：所有背景像素中被主色覆盖的比例
    min_dists = np.min(distances, axis=1)
    covered = (min_dists < threshold).sum()
    print(f"\n{'=' * 70}")
    print(f"总结:")
    print(f"  背景像素总数: {len(bg_pixels)}")
    print(f"  被主色覆盖 (Lab<25): {covered} ({covered/len(bg_pixels):.1%})")
    print(f"  未被覆盖: {len(bg_pixels)-covered} ({1-covered/len(bg_pixels):.1%})")

    # 亮度分布
    L = bg_pixels[:, 0]
    print(f"\n  背景亮度 (L) 分布:")
    print(f"    L<5 (近黑): {(L < 5).sum()} ({(L < 5).mean():.1%})")
    print(f"    5<=L<15 (暗): {((L >= 5) & (L < 15)).sum()} ({((L >= 5) & (L < 15)).mean():.1%})")
    print(f"    15<=L<30 (中暗): {((L >= 15) & (L < 30)).sum()} ({((L >= 15) & (L < 30)).mean():.1%})")
    print(f"    30<=L<50 (中): {((L >= 30) & (L < 50)).sum()} ({((L >= 30) & (L < 50)).mean():.1%})")
    print(f"    L>=50 (亮): {(L >= 50).sum()} ({(L >= 50).mean():.1%})")


if __name__ == "__main__":
    image_path = r"D:\Code\GraphColor\extracted_imgs\imgs\102548643_p0.jpg"
    json_path = r"D:\Code\GraphColor\outputs\results.json"
    analyze_image_background(image_path, json_path)
