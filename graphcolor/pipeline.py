"""
批量处理管线。

编排整个流程：
  加载 → resize → Lab转换 → 主体/背景分离 → 各自聚类 → 评分 → JSON/终端输出

本模块是 `graphcolor` 教师模型的"门面":
  - GraphColorPipeline.process()  处理单张图片
  - GraphColorPipeline.process_batch()  处理多张图片(可选多进程)
  - process_batch()  便捷函数,直接拿到 dict 列表
  - image_result_to_dict()  把 ImageResult 转成可 JSON 序列化的 dict
  - main()  命令行入口(`python -m graphcolor.pipeline ...`)
"""
import json
import time
from pathlib import Path
from typing import List, Optional

import numpy as np

import cv2

from .preprocess import load_image, resize_to_max, bgr_to_lab
from .segment import ForegroundSegmenter, NeuralSegmenter, SegmentResult, _make_result
from .cluster import LabClusterer, ClusterResult
from .scoring import (
    ClusterScorer, MainColor, RegionResult, ImageResult, analyze_region
)


DEFAULT_CONFIG = {
    # 预处理
    "max_size": 512,
    # 分割（3个核心参数，控制主体识别的激进程度）
    "chroma_threshold": 12.0,       # 色度阈值：越低→识别越激进，更多彩色区域被纳入主体
    "min_contour_area_ratio": 0.02,  # 最小轮廓面积比：越低→保留更小的轮廓细节
    "foreground_score_threshold": 0.5, # 前景得分阈值：越低→主体识别越激进
    # 空洞检测（GrabCut 精修用，控制环状主体内部空洞的识别）
    "bg_colors_n": 1,                 # 用于空洞检测的背景主色数量（取评分Top N）
    "hole_distance_threshold": 15.0,  # 空洞距离阈值：前景像素到背景主色的BGR欧氏距离上限
    "hole_min_area_ratio": 0.002,     # 空洞最小面积比：低于此值不视为空洞
    "grabcut_iterations": 1,          # GrabCut 迭代次数：越多→精修越激进
    # 聚类
    "n_clusters_foreground": 10,
    "n_clusters_background": 6,
    "color_weight": 1.0,
    "lightness_weight": 0.5,
    "a_boost": 1.1,
    "batch_size": 1024,
    # 评分
    "weight_count": 0.5,
    "weight_variance": 0.08,
    "weight_center": 0.08,
    "weight_chroma": 0.14,
    "weight_bilinear": 0.2,           # count*salience 双线性项 (0=退化为加算)
    "min_score_display": 0.05,
    # 视觉显著性参数
    "salience_a_weight": 1.2,          # 色度计算中 a* 的加权: chroma=sqrt(a_weight*a²+b²)
    "salience_chroma_logit": 1.0,      # 色度/亮度权重比值 logit, sigmoid后 w_chroma=w_L=0.5
    # 肤色惩罚 (人像图片中让肤色以外的颜色更突出)
    #   - off: 关闭肤色惩罚
    #   - fixed: 使用 DEFAULT_SKIN_REF_LAB / DEFAULT_SKIN_SIGMA 作为参考
    #            (固定 Lab 空间，参考中心不随图像变化)
    "skin_penalty_mode": "fixed",
    "skin_penalty_max": 0.1,          # 肤色惩罚上限 (0~1)，默认 10% 的最大折扣
    "skin_penalty_apply_bg": False,    # 是否对背景区域也施加肤色惩罚 (默认仅前景)
}


class GraphColorPipeline:
    """
    主色提取管线
    """

    def __init__(self, config: Optional[dict] = None):
        self.config = {**DEFAULT_CONFIG, **(config or {})}

        # 分割器选择：默认使用 NeuralSegmenter（深度学习），
        # 可通过 config["use_grabcut"] = True 切换回传统 GrabCut 方案
        if self.config.get("use_grabcut", False):
            self.segmenter = ForegroundSegmenter(
                chroma_threshold=self.config["chroma_threshold"],
                min_contour_area_ratio=self.config["min_contour_area_ratio"],
                foreground_score_threshold=self.config.get("foreground_score_threshold", 0.55),
            )
        else:
            self.segmenter = NeuralSegmenter(
                model_name=self.config.get("neural_model", "isnet-general-use"),
            )

        self.fg_clusterer = LabClusterer(
            n_clusters=self.config["n_clusters_foreground"],
            color_weight=self.config["color_weight"],
            lightness_weight=self.config["lightness_weight"],
            a_boost=self.config.get("a_boost", 1.1),
            batch_size=self.config["batch_size"],
        )

        self.bg_clusterer = LabClusterer(
            n_clusters=self.config["n_clusters_background"],
            color_weight=self.config["color_weight"],
            lightness_weight=self.config["lightness_weight"],
            a_boost=self.config.get("a_boost", 1.1),
            batch_size=self.config["batch_size"],
        )

        self.scorer = ClusterScorer(
            weight_count=self.config["weight_count"],
            weight_variance=self.config["weight_variance"],
            weight_center=self.config["weight_center"],
            weight_chroma=self.config["weight_chroma"],
            weight_bilinear=self.config.get("weight_bilinear", 0.0),
            min_score_display=self.config["min_score_display"],
            salience_a_weight=self.config.get("salience_a_weight", 1.2),
            salience_chroma_logit=self.config.get("salience_chroma_logit", 0.0),
            skin_penalty_max=self._resolve_skin_penalty_max(),
        )

    def _resolve_skin_penalty_max(self) -> float:
        """根据 skin_penalty_mode 决定 ClusterScorer 上的默认惩罚值。

        - "off": 0（完全关闭）
        - "fixed": skin_penalty_max
        - 其他（兼容旧值）: 视为 fixed
        """
        mode = self.config.get("skin_penalty_mode", "off")
        if mode == "off":
            return 0.0
        return float(self.config.get("skin_penalty_max", 0.10))

    def warmup(self):
        """预加载分割模型，避免首张图片处理耗时过长。"""
        if hasattr(self.segmenter, "warmup"):
            self.segmenter.warmup()

    def process(self, image_path: str) -> ImageResult:
        """
        完整工作流（对于每张图片）：
        1. 获取图片（和相应的512*512压缩图片）
        2. 进行初步rembg主体提取
        3. 计算获取背景主色
        4. 利用背景主色和GrabCut移除主体可能存在的空洞
        5. 计算获取主体主色
        6. 输出该张图片的结果

        Args:
            image_path: 图片路径

        Returns:
            ImageResult
        """
        path = Path(image_path)

        # 步骤1: 加载 & 缩放
        original_bgr, alpha = load_image(str(path))
        bgr = resize_to_max(original_bgr, self.config["max_size"])
        alpha_small = None
        if alpha is not None:
            alpha_small = resize_to_max(alpha, self.config["max_size"], cv2.INTER_AREA)
        h, w = bgr.shape[:2]

        # 步骤2: BGR -> Lab
        lab = bgr_to_lab(bgr)

        # 步骤3: rembg初步主体提取
        if hasattr(self.segmenter, "extract_mask"):
            # NeuralSegmenter: 使用新的流式API
            neural_mask = self.segmenter.extract_mask(bgr, alpha_small)
            if neural_mask is None:
                # 提取失败，回退到传统segment
                seg_result = self.segmenter.segment(lab, bgr_img=bgr, alpha_mask=alpha_small)
            else:
                # 步骤4: 计算背景主色（完整聚类+评分）
                bg_mask_initial = ~neural_mask
                bg_pixels = lab[bg_mask_initial]
                yy, xx = np.mgrid[:h, :w]
                coords = np.stack([yy, xx], axis=2)
                bg_coords = coords[bg_mask_initial]

                bg_result, bg_skin_info = analyze_region(
                    "background", bg_pixels, bg_coords,
                    self.bg_clusterer, self.scorer, h, w,
                )

                # 步骤5: 利用背景主色和GrabCut精修主体mask（移除空洞）
                if hasattr(self.segmenter, "refine_mask"):
                    # 取背景主色（BGR格式），传入refine_mask
                    bg_colors_bgr = np.array(
                        [mc.rgb[::-1] for mc in bg_result.main_colors[:self.config["bg_colors_n"]]],
                        dtype=np.float32
                    ) if bg_result.main_colors else np.zeros((0, 3), dtype=np.float32)

                    refined_mask = self.segmenter.refine_mask(
                        bgr, neural_mask, bg_colors_bgr,
                        hole_distance_threshold=self.config["hole_distance_threshold"],
                        hole_min_area_ratio=self.config["hole_min_area_ratio"],
                        grabcut_iterations=self.config["grabcut_iterations"],
                    )
                    if refined_mask is None:
                        refined_mask = neural_mask
                else:
                    refined_mask = neural_mask

                # 步骤6: 计算前景主色
                fg_mask = refined_mask
                fg_pixels = lab[fg_mask]
                fg_coords = coords[fg_mask]

                fg_result, fg_skin_info = analyze_region(
                    "foreground", fg_pixels, fg_coords,
                    self.fg_clusterer, self.scorer, h, w,
                )

                seg_result = _make_result(fg_mask, f"neural_{self.segmenter.model_name}")
                seg_result.initial_mask = neural_mask  # 保存首次提取的mask用于可视化

                return ImageResult(
                    image_path=str(path),
                    foreground=fg_result,
                    background=bg_result,
                    segment_method=seg_result.method,
                    seg_result=seg_result,
                    skin_info={'fg': fg_skin_info, 'bg': bg_skin_info},
                )
        else:
            # ForegroundSegmenter: 使用传统方式
            seg_result = self.segmenter.segment(lab, bgr_img=bgr, alpha_mask=alpha_small)

        # 步骤4: 提取各区域像素（回退路径）
        fg_mask = seg_result.foreground_mask
        bg_mask = seg_result.background_mask

        # 构建像素坐标
        yy, xx = np.mgrid[:h, :w]
        coords = np.stack([yy, xx], axis=2)

        fg_pixels = lab[fg_mask]
        fg_coords = coords[fg_mask]
        bg_pixels = lab[bg_mask]
        bg_coords = coords[bg_mask]

        # 步骤5: 前景分析
        fg_result, fg_skin_info = analyze_region(
            "foreground", fg_pixels, fg_coords,
            self.fg_clusterer, self.scorer, h, w,
        )

        # 步骤6: 背景分析
        bg_result, bg_skin_info = analyze_region(
            "background", bg_pixels, bg_coords,
            self.bg_clusterer, self.scorer, h, w,
        )

        return ImageResult(
            image_path=str(path),
            foreground=fg_result,
            background=bg_result,
            segment_method=seg_result.method,
            seg_result=seg_result,
            skin_info={'fg': fg_skin_info, 'bg': bg_skin_info},
        )

    def process_batch(self, image_paths: List[str],
                      verbose: bool = True,
                      workers: int = 1) -> List[ImageResult]:
        """
        批量处理多张图片。

        Args:
            image_paths: 图片路径列表
            verbose: 是否打印进度
            workers: 并行工作进程数 (1=单线程, >1=多进程)

        Returns:
            ImageResult列表
        """
        # Warmup: 预加载分割模型（不计入计时）
        self.warmup()

        if workers <= 1:
            return self._process_batch_sequential(image_paths, verbose)

        return self._process_batch_parallel(image_paths, verbose, workers)

    def _process_batch_sequential(self, image_paths: List[str],
                                   verbose: bool) -> List[ImageResult]:
        """单线程顺序处理"""
        results = []
        total = len(image_paths)
        errors = 0
        t_start = time.perf_counter()

        for i, path in enumerate(image_paths):
            try:
                result = self.process(path)
                results.append(result)

                if verbose:
                    print(f"  [{i + 1}/{total}] OK  {Path(path).name}", flush=True)
            except Exception as e:
                errors += 1
                if verbose:
                    print(f"  [{i + 1}/{total}] FAIL {Path(path).name}: {e}", flush=True)

        t_total = time.perf_counter() - t_start
        if verbose:
            ok = len(results)
            print(f"\n  完成: {ok}/{total} 成功, 总耗时 {t_total:.1f}s"
                  f" (均 {t_total/max(1, ok):.1f}s/张)")

        return results

    def _process_batch_parallel(self, image_paths: List[str],
                                 verbose: bool,
                                 workers: int) -> List[ImageResult]:
        """多进程并行处理"""
        from concurrent.futures import ProcessPoolExecutor, as_completed

        results = [None] * len(image_paths)
        errors = 0
        t_start = time.perf_counter()

        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(self._process_single, path, self.config): (i, path)
                for i, path in enumerate(image_paths)
            }
            done_count = 0
            total = len(image_paths)
            for future in as_completed(futures):
                idx, path = futures[future]
                done_count += 1
                try:
                    result = future.result()
                    results[idx] = result

                    if verbose:
                        print(f"  [{done_count}/{total}] OK  {Path(path).name}", flush=True)
                except Exception as e:
                    errors += 1
                    if verbose:
                        print(f"  [{done_count}/{total}] FAIL {Path(path).name}: {e}", flush=True)

        t_total = time.perf_counter() - t_start
        results = [r for r in results if r is not None]
        if verbose:
            ok = len(results)
            print(f"\n  完成: {ok}/{total} 成功 ({workers}进程并行), "
                  f"总耗时 {t_total:.1f}s (均 {t_total/max(1, ok):.1f}s/张)")

        return results

    @staticmethod
    def _process_single(image_path: str, config: dict) -> ImageResult:
        """多进程工作函数：创建独立管线处理单张图片"""
        p = GraphColorPipeline(config)
        p.warmup()
        return p.process(image_path)


def image_result_to_dict(result: ImageResult) -> dict:
    """
    将 ImageResult 转换为可 JSON 序列化的字典。

    序列化时只保留 image / segment_method / foreground / background 四个字段;
    `MainColor` 中的 lab / rgb / hex / score / proportion 全部转成纯 python
    类型(避免 numpy.float64 / numpy.int64 在 json.dump 时报错)。
    """
    def color_to_dict(c: MainColor) -> dict:
        return {
            "lab": [round(float(c.lab[0]), 1),
                    round(float(c.lab[1]), 1),
                    round(float(c.lab[2]), 1)],
            "rgb": [int(c.rgb[0]), int(c.rgb[1]), int(c.rgb[2])],
            "hex": c.hex_color,
            "score": round(c.score, 4),
            "proportion": round(c.proportion, 4),
        }

    def region_to_dict(r: RegionResult) -> dict:
        return {
            "dominant_color": color_to_dict(r.dominant_color),
            "main_colors": [color_to_dict(c) for c in r.main_colors],
        }

    return {
        "image": result.image_path,
        "segment_method": result.segment_method,
        "foreground": region_to_dict(result.foreground),
        "background": region_to_dict(result.background),
    }


def process_batch(image_paths: List[str],
                  config: Optional[dict] = None,
                  output_json: Optional[str] = None,
                  verbose: bool = True,
                  workers: int = 1) -> List[dict]:
    """
    便捷的批量处理入口函数。

    Args:
        image_paths: 图片路径列表
        config: 配置覆盖字典
        output_json: 若有值，将结果写入此JSON文件
        verbose: 是否打印进度
        workers: 并行工作进程数 (1=单线程, >1=多进程)

    Returns:
        字典列表（可直接序列化为JSON）
    """
    pipeline = GraphColorPipeline(config)

    results = pipeline.process_batch(image_paths, verbose=verbose, workers=workers)
    dict_results = [image_result_to_dict(r) for r in results]

    if output_json:
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(dict_results, f, ensure_ascii=False, indent=2)
        if verbose:
            print(f"\n结果已保存至: {output_json}")

    return dict_results


# ─── CLI 入口 ────────────────────────────────────────────────────────────

def main():
    """pipeline.py 的命令行入口"""
    import argparse

    parser = argparse.ArgumentParser(
        description="GraphColor 主色提取管线",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python -m graphcolor.pipeline image.png
  python -m graphcolor.pipeline img1.jpg img2.jpg --output results.json
  python -m graphcolor.pipeline "images/*.png" --max-size 256
        """
    )
    parser.add_argument(
        "images", nargs="+",
        help="图片路径，支持通配符（如 *.png）"
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="输出JSON文件路径"
    )
    parser.add_argument(
        "--max-size", type=int, default=512,
        help="压缩后图片最长边像素数 (默认: 512)"
    )
    parser.add_argument(
        "--clusters-fg", type=int, default=10,
        help="前景聚类数 (默认: 10)"
    )
    parser.add_argument(
        "--clusters-bg", type=int, default=6,
        help="背景聚类数 (默认: 6)"
    )
    parser.add_argument(
        "--workers", "-j", type=int, default=1,
        help="并行工作进程数 (默认: 1=单线程)"
    )
    args = parser.parse_args()

    # 展开通配符
    import glob
    image_paths = []
    for p in args.images:
        expanded = glob.glob(p, recursive=True)
        if expanded:
            image_paths.extend(expanded)
        else:
            image_paths.append(p)

    if not image_paths:
        print("错误: 未找到任何图片文件")
        return

    print(f"找到 {len(image_paths)} 张图片")

    config = {
        "max_size": args.max_size,
        "n_clusters_foreground": args.clusters_fg,
        "n_clusters_background": args.clusters_bg,
    }

    results = process_batch(
        image_paths,
        config=config,
        output_json=args.output,
        verbose=True,
        workers=args.workers
    )

    print(f"\n成功处理 {len(results)} 张图片")


if __name__ == "__main__":
    main()
