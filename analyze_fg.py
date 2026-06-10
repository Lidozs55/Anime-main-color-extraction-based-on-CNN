#!/usr/bin/env python
"""
分析图片前景的颜色构成，与JSON结果中的前景主色进行对比。
支持多图（100870710 有 p0/p1/p2/p3 四张）。
"""
import json
import numpy as np
import cv2
from pathlib import Path
from scipy.spatial.distance import cdist

from graphcolor.preprocess import load_image, bgr_to_lab, lab_to_rgb_for_display
from graphcolor.segment import NeuralSegmenter


def analyze_foreground(image_path: str, json_path: str = "outputs/results.json"):
    # 加载JSON结果
    with open(json_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    stem = Path(image_path).stem
    target = None
    for r in results:
        if stem in r["image"]:
            target = r
            break

    if target is None:
        print(f"未找到包含 {stem} 的结果")
        return

    print("=" * 70)
    print(f"分析图片: {target['image']}")
    print(f"分割方法: {target['segment_method']}")
    print("=" * 70)

    # 打印JSON中的前景主色
    fg_colors = target["foreground"]["main_colors"]
    print(f"\nJSON中的前景主色 ({len(fg_colors)}个):")
    for i, c in enumerate(fg_colors):
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
    fg_mask = seg_result.foreground_mask
    fg_pixels = lab[fg_mask]
    print(f"前景占比: {seg_result.foreground_ratio:.1%} | 背景占比: {1-seg_result.foreground_ratio:.1%}")
    print(f"前景像素数: {len(fg_pixels)}")

    # 前景像素的RGB和Lab统计
    fg_rgb = lab_to_rgb_for_display(fg_pixels)
    print(f"\n前景像素 RGB 统计:")
    print(f"  均值:   R={fg_rgb[:, 0].mean():.1f}, G={fg_rgb[:, 1].mean():.1f}, B={fg_rgb[:, 2].mean():.1f}")
    print(f"  中位数: R={np.median(fg_rgb[:, 0]):.1f}, G={np.median(fg_rgb[:, 1]):.1f}, B={np.median(fg_rgb[:, 2]):.1f}")
    print(f"  标准差: R={fg_rgb[:, 0].std():.1f}, G={fg_rgb[:, 1].std():.1f}, B={fg_rgb[:, 2].std():.1f}")

    print(f"\n前景像素 Lab 统计:")
    print(f"  均值:   L={fg_pixels[:, 0].mean():.1f}, a={fg_pixels[:, 1].mean():.1f}, b={fg_pixels[:, 2].mean():.1f}")
    print(f"  中位数: L={np.median(fg_pixels[:, 0]):.1f}, a={np.median(fg_pixels[:, 1]):.1f}, b={np.median(fg_pixels[:, 2]):.1f}")
    print(f"  标准差: L={fg_pixels[:, 0].std():.1f}, a={fg_pixels[:, 1].std():.1f}, b={fg_pixels[:, 2].std():.1f}")

    # 分析各前景主色的匹配情况
    print(f"\n各主色匹配分析 (基于Lab空间距离):")
    fg_centers_lab = np.array([c["lab"] for c in fg_colors])

    distances = cdist(fg_pixels, fg_centers_lab, metric='euclidean')
    nearest = np.argmin(distances, axis=1)
    threshold = 25.0

    for i, c in enumerate(fg_colors):
        mask = nearest == i
        count = mask.sum()
        proportion = count / len(fg_pixels)
        matched_mask = distances[:, i] < threshold
        matched_count = matched_mask.sum()
        matched_proportion = matched_count / len(fg_pixels)

        if matched_count > 0:
            matched_pixels = fg_pixels[matched_mask]
            matched_rgb = fg_rgb[matched_mask]
            print(f"\n  [{i}] {c['hex']} (JSON proportion={c['proportion']:.4f}):")
            print(f"      最近归属: {count} 像素 ({proportion:.4f})")
            print(f"      Lab<25匹配: {matched_count} 像素 ({matched_proportion:.4f})")
            print(f"      匹配区域 RGB 均值: R={matched_rgb[:, 0].mean():.1f}, G={matched_rgb[:, 1].mean():.1f}, B={matched_rgb[:, 2].mean():.1f}")
            print(f"      匹配区域 RGB 中位数: R={np.median(matched_rgb[:, 0]):.0f}, G={np.median(matched_rgb[:, 1]):.0f}, B={np.median(matched_rgb[:, 2]):.0f}")
            print(f"      匹配区域 Lab 均值: L={matched_pixels[:, 0].mean():.1f}, a={matched_pixels[:, 1].mean():.1f}, b={matched_pixels[:, 2].mean():.1f}")
        else:
            print(f"\n  [{i}] {c['hex']}: 无匹配像素")

    # 整体覆盖度
    min_dists = np.min(distances, axis=1)
    covered = (min_dists < threshold).sum()
    print(f"\n{'=' * 70}")
    print(f"总结:")
    print(f"  前景像素总数: {len(fg_pixels)}")
    print(f"  被主色覆盖 (Lab<25): {covered} ({covered/len(fg_pixels):.1%})")
    print(f"  未被覆盖: {len(fg_pixels)-covered} ({1-covered/len(fg_pixels):.1%})")

    # 亮度分布
    L = fg_pixels[:, 0]
    print(f"\n  前景亮度 (L) 分布:")
    print(f"    L<15 (暗): {(L < 15).sum()} ({(L < 15).mean():.1%})")
    print(f"    15<=L<30 (中暗): {((L >= 15) & (L < 30)).sum()} ({((L >= 15) & (L < 30)).mean():.1%})")
    print(f"    30<=L<50 (中): {((L >= 30) & (L < 50)).sum()} ({((L >= 30) & (L < 50)).mean():.1%})")
    print(f"    50<=L<70 (中亮): {((L >= 50) & (L < 70)).sum()} ({((L >= 50) & (L < 70)).mean():.1%})")
    print(f"    L>=70 (亮): {(L >= 70).sum()} ({(L >= 70).mean():.1%})")

    # 色度分布
    a = fg_pixels[:, 1]
    b = fg_pixels[:, 2]
    chroma = np.sqrt(a**2 + b**2)
    print(f"\n  前景色度 (C*) 分布:")
    print(f"    C*<10 (近灰): {(chroma < 10).sum()} ({(chroma < 10).mean():.1%})")
    print(f"    10<=C*<25 (低): {((chroma >= 10) & (chroma < 25)).sum()} ({((chroma >= 10) & (chroma < 25)).mean():.1%})")
    print(f"    25<=C*<50 (中): {((chroma >= 25) & (chroma < 50)).sum()} ({((chroma >= 25) & (chroma < 50)).mean():.1%})")
    print(f"    C*>=50 (高): {(chroma >= 50).sum()} ({(chroma >= 50).mean():.1%})")

    # 对比分析：主色与聚类中心的关系
    print(f"\n{'=' * 70}")
    print(f"主色中心 vs 匹配区域均值对比:")
    for i, c in enumerate(fg_colors):
        matched_mask = distances[:, i] < threshold
        if matched_mask.sum() > 0:
            matched_lab = fg_pixels[matched_mask]
            center_lab = np.array(c["lab"])
            mean_lab = matched_lab.mean(axis=0)
            delta_lab = mean_lab - center_lab
            delta_e = np.sqrt(np.sum(delta_lab**2))
            print(f"  [{i}] {c['hex']}: 中心={center_lab.round(1)}, 均值={mean_lab.round(1)}, ΔE={delta_e:.1f}")


if __name__ == "__main__":
    json_path = r"D:\Code\GraphColor\outputs\results.json"
    # 100870710 有四张图，都分析一下
    imgs_dir = r"D:\Code\GraphColor\extracted_imgs\imgs"
    for p in ["p0", "p1", "p2", "p3"]:
        image_path = f"{imgs_dir}/100870710_{p}.png"
        if Path(image_path).exists():
            analyze_foreground(image_path, json_path)
            print("\n\n")
