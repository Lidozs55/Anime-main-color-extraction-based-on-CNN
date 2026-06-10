#!/usr/bin/env python
"""
GraphColor 管线测试脚本。

生成合成测试图片并验证主色提取流程。
"""
import sys
import os
import tempfile
import json
from pathlib import Path

import numpy as np
from PIL import Image

# 确保可以导入graphcolor
sys.path.insert(0, str(Path(__file__).resolve().parent))
from graphcolor import GraphColorPipeline, process_batch
from graphcolor.pipeline import image_result_to_dict


def create_test_images(output_dir: str):
    """
    生成多张合成测试图片：

    1. test_simple.png  - 红球在白色背景上
    2. test_anime.png   - 模拟动漫风格的彩色主体在渐变背景上
    3. test_multi.png   - 多个彩色物体在灰色背景上
    4. test_white_fg.png - 浅色主体在深色背景上（测试亮度权重）
    """
    images = {}

    # 1. 红球在白色背景上
    img = Image.new("RGB", (256, 256), (255, 255, 255))
    pixels = np.array(img)
    yy, xx = np.ogrid[:256, :256]
    center = np.sqrt((yy - 128) ** 2 + (xx - 128) ** 2)
    circle = center <= 80
    pixels[circle] = (220, 60, 60)  # 红色
    # 加一些渐变
    highlight = center <= 40
    pixels[highlight] = (240, 80, 80)
    shadow_ring = (center > 60) & (center <= 80)
    pixels[shadow_ring] = (180, 40, 40)
    images["test_simple.png"] = Image.fromarray(pixels)

    # 2. 模拟动漫风格
    img2 = Image.new("RGB", (300, 300), (200, 220, 240))  # 浅蓝灰背景
    pixels2 = np.array(img2)
    # 中心人物（动漫风格配色）
    yy2, xx2 = np.ogrid[:300, :300]
    body_mask = (np.abs(yy2 - 150) <= 100) & (np.abs(xx2 - 150) <= 50)
    pixels2[body_mask] = (80, 140, 220)  # 蓝色衣服

    head_mask = (np.sqrt((yy2 - 80) ** 2 + (xx2 - 150) ** 2) <= 40)
    pixels2[head_mask] = (255, 220, 180)  # 肤色

    hair_mask = (np.sqrt((yy2 - 70) ** 2 + (xx2 - 150) ** 2) <= 35) & ~head_mask
    pixels2[hair_mask] = (50, 50, 50)  # 黑色头发

    # 加一些红色装饰
    ribbon_mask = (np.abs(yy2 - 90) <= 8) & (np.abs(xx2 - 110) <= 15)
    pixels2[ribbon_mask] = (220, 50, 50)
    images["test_anime.png"] = Image.fromarray(pixels2)

    # 3. 多个彩色物体在灰色背景上
    img3 = Image.new("RGB", (256, 256), (180, 180, 180))
    pixels3 = np.array(img3)
    # 红方块
    pixels3[40:100, 30:90] = (220, 60, 60)
    # 绿圆
    yy3, xx3 = np.ogrid[:256, :256]
    green_circle = np.sqrt((yy3 - 140) ** 2 + (xx3 - 180) ** 2) <= 50
    pixels3[green_circle] = (60, 200, 80)
    # 蓝三角（近似）
    blue_tri = (xx3 >= 30) & (xx3 <= 100) & (yy3 >= 150) & (yy3 <= 220) & (xx3 >= yy3 - 100)
    pixels3[blue_tri] = (40, 80, 200)
    images["test_multi.png"] = Image.fromarray(pixels3)

    # 4. 浅色主体在深色背景
    img4 = Image.new("RGB", (256, 256), (30, 30, 50))
    pixels4 = np.array(img4)
    center4 = np.sqrt((yy - 128) ** 2 + (xx - 128) ** 2)
    circle4 = center4 <= 90
    pixels4[circle4] = (240, 235, 220)  # 米白主体
    pixels4[center4 <= 60] = (250, 245, 230)
    pixels4[(center4 > 60) & (center4 <= 90)] = (220, 210, 190)
    images["test_white_fg.png"] = Image.fromarray(pixels4)

    # 保存所有图片
    for name, img in images.items():
        path = os.path.join(output_dir, name)
        img.save(path)
        print(f"  生成: {path} ({img.size[0]}x{img.size[1]})")

    return list(images.keys())


def run_test():
    """执行测试"""
    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"创建临时测试目录: {tmpdir}")
        print("生成测试图片...")
        filenames = create_test_images(tmpdir)
        image_paths = [os.path.join(tmpdir, f) for f in filenames]

        print("\n" + "=" * 60)
        print("测试1: 默认配置管线")
        print("=" * 60)
        pipeline = GraphColorPipeline()
        for path in image_paths:
            result = pipeline.process(path)
            d = image_result_to_dict(result)
            fg = d["foreground"]
            bg = d["background"]
            print(f"\n  图片: {Path(path).name}")
            print(f"  分割方法: {result.segment_method}")
            print(f"  前景主色: {fg['dominant_color']['hex']} "
                  f"(score={fg['dominant_color']['score']:.4f}, "
                  f"rgb={fg['dominant_color']['rgb']})")
            print(f"  前景主色列表:")
            for c in fg["main_colors"][:3]:
                print(f"    {c['hex']} score={c['score']:.4f} prop={c['proportion']:.3f}")
            print(f"  背景主色: {bg['dominant_color']['hex']} "
                  f"(score={bg['dominant_color']['score']:.4f})")
            if bg["main_colors"]:
                print(f"  背景主色列表:")
                for c in bg["main_colors"][:3]:
                    print(f"    {c['hex']} score={c['score']:.4f} prop={c['proportion']:.3f}")

        print("\n" + "=" * 60)
        print("测试2: 快速模式 (lightness_weight=0, 忽略亮度)")
        print("=" * 60)
        pipeline_fast = GraphColorPipeline({
            "lightness_weight": 0.0,
            "n_clusters_foreground": 6,
            "n_clusters_background": 4,
        })
        result = pipeline_fast.process(image_paths[0])
        d = image_result_to_dict(result)
        print(f"  图片: {Path(image_paths[0]).name}")
        print(f"  前景主色: {d['foreground']['dominant_color']['hex']}")
        print(f"  背景主色: {d['background']['dominant_color']['hex']}")

        print("\n" + "=" * 60)
        print("测试3: 批量处理 + JSON输出 + 结果图")
        print("=" * 60)
        json_path = os.path.join(tmpdir, "test_results.json")
        preview_dir = os.path.join(tmpdir, "previews")
        results = process_batch(
            image_paths,
            output_json=json_path,
            output_preview_dir=preview_dir,
            verbose=True
        )

        # 验证JSON可读
        with open(json_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        assert len(loaded) == len(image_paths), f"JSON结果数 {len(loaded)} 不匹配 {len(image_paths)}"
        preview_files = list(Path(preview_dir).glob("*_graphcolor.png"))
        assert len(preview_files) == len(image_paths), (
            f"结果图数量 {len(preview_files)} 不匹配 {len(image_paths)}"
        )
        print(f"\n✓ JSON输出验证通过: {json_path}")
        print(f"✓ 结果图输出验证通过: {preview_dir}")
        print(f"✓ 成功处理 {len(results)} 张图片")

    print("\n" + "=" * 60)
    print("全部测试通过!")
    print("=" * 60)


if __name__ == "__main__":
    run_test()
