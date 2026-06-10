#!/usr/bin/env python
"""
速度测试：单线程 vs 多线程/多进程性能对比。

用法:
    python speed_test.py <图片路径...>
    python speed_test.py "images/*.png" --max-images 20 --workers 4
"""
import argparse
import glob
import os
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

# 设置环境变量以最大化性能
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
os.environ["OPENBLAS_NUM_THREADS"] = "8"
os.environ["NUMEXPR_NUM_THREADS"] = "8"

from graphcolor.pipeline import GraphColorPipeline, image_result_to_dict
from graphcolor.preprocess import load_image, resize_to_max, bgr_to_lab
from graphcolor.segment import NeuralSegmenter

IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}


def get_image_paths(patterns, max_images=None):
    """展开图片路径"""
    paths = []
    for p in patterns:
        expanded = glob.glob(p, recursive=True)
        if expanded:
            paths.extend(expanded)
        else:
            paths.append(p)

    result = []
    for p in paths:
        path = Path(p)
        if path.suffix.lower() in IMAGE_EXTS and path.exists():
            result.append(str(path.resolve()))
        elif path.is_dir():
            for child in path.rglob("*"):
                if child.suffix.lower() in IMAGE_EXTS:
                    result.append(str(child.resolve()))

    if max_images:
        result = result[:max_images]
    return result


def process_single_image(image_path, config=None):
    """处理单张图片（用于多进程）"""
    pipeline = GraphColorPipeline(config)
    pipeline.warmup()
    start = time.perf_counter()
    result = pipeline.process(image_path)
    elapsed = time.perf_counter() - start
    return {
        "path": image_path,
        "time": elapsed,
        "method": result.segment_method,
        "fg_color": result.foreground.dominant_color.hex_color,
        "bg_color": result.background.dominant_color.hex_color,
    }


def run_benchmark(image_paths, config, mode="single", workers=4):
    """运行基准测试"""
    print(f"\n{'='*60}")
    print(f"测试模式: {mode} | 图片数: {len(image_paths)} | 工作线程: {workers}")
    print(f"{'='*60}")

    # Warmup first
    warmup_pipeline = GraphColorPipeline(config)
    warmup_pipeline.warmup()
    print(f"Warmup 完成")

    times = []
    results = []

    t_start = time.perf_counter()

    if mode == "single":
        for i, path in enumerate(image_paths):
            t0 = time.perf_counter()
            result = process_single_image(path, config)
            elapsed = time.perf_counter() - t0
            times.append(elapsed)
            results.append(result)
            print(f"  [{i+1}/{len(image_paths)}] OK {Path(path).name} ({elapsed:.2f}s)")

    elif mode == "thread":
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(process_single_image, p, config): p
                       for p in image_paths}
            for i, future in enumerate(as_completed(futures)):
                result = future.result()
                times.append(result["time"])
                results.append(result)
                print(f"  [{i+1}/{len(image_paths)}] OK {Path(result['path']).name} ({result['time']:.2f}s)")

    elif mode == "process":
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(process_single_image, p, config): p
                       for p in image_paths}
            for i, future in enumerate(as_completed(futures)):
                result = future.result()
                times.append(result["time"])
                results.append(result)
                print(f"  [{i+1}/{len(image_paths)}] OK {Path(result['path']).name} ({result['time']:.2f}s)")

    t_total = time.perf_counter() - t_start

    # 统计
    avg_time = np.mean(times)
    median_time = np.median(times)
    min_time = np.min(times)
    max_time = np.max(times)
    throughput = len(image_paths) / t_total

    print(f"\n--- {mode} 模式统计 ---")
    print(f"  总耗时: {t_total:.2f}s")
    print(f"  吞吐量: {throughput:.2f} 张/秒")
    print(f"  平均: {avg_time:.2f}s | 中位数: {median_time:.2f}s")
    print(f"  最快: {min_time:.2f}s | 最慢: {max_time:.2f}s")

    return {
        "mode": mode,
        "workers": workers,
        "total_time": t_total,
        "throughput": throughput,
        "avg_time": avg_time,
        "median_time": median_time,
        "min_time": min_time,
        "max_time": max_time,
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(description="GraphColor 速度测试")
    parser.add_argument("images", nargs="+", help="图片路径/目录/通配符")
    parser.add_argument("--max-images", type=int, default=10,
                        help="最多测试图片数 (默认: 10)")
    parser.add_argument("--max-size", type=int, default=512, help="最大尺寸")
    parser.add_argument("--workers", type=int, default=4, help="工作线程/进程数")
    parser.add_argument("--skip-benchmark", action="store_true",
                        help="跳过对比测试，仅用单线程运行")
    args = parser.parse_args()

    image_paths = get_image_paths(args.images, args.max_images)
    if not image_paths:
        print("错误: 未找到任何图片")
        return

    print(f"找到 {len(image_paths)} 张图片")

    config = {
        "max_size": args.max_size,
        "n_clusters_foreground": 10,
        "n_clusters_background": 6,
        "color_weight": 3.0,
        "lightness_weight": 0.2,
    }

    # 单线程基准
    single_result = run_benchmark(image_paths, config, mode="single")

    if args.skip_benchmark:
        return

    # 多线程测试
    thread_result = run_benchmark(image_paths, config, mode="thread",
                                  workers=args.workers)

    # 多进程测试
    process_result = run_benchmark(image_paths, config, mode="process",
                                   workers=args.workers)

    # 对比总结
    print(f"\n{'='*60}")
    print("性能对比总结")
    print(f"{'='*60}")
    print(f"  单线程: {single_result['throughput']:.2f} 张/秒 "
          f"(平均 {single_result['avg_time']:.2f}s)")
    print(f"  多线程({args.workers}): {thread_result['throughput']:.2f} 张/秒 "
          f"(平均 {thread_result['avg_time']:.2f}s) "
          f"→ {thread_result['throughput']/single_result['throughput']:.1f}x")
    print(f"  多进程({args.workers}): {process_result['throughput']:.2f} 张/秒 "
          f"(平均 {process_result['avg_time']:.2f}s) "
          f"→ {process_result['throughput']/single_result['throughput']:.1f}x")


if __name__ == "__main__":
    main()
