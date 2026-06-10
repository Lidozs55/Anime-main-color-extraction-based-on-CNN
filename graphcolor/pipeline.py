"""
批量处理管线。

编排整个流程：
  加载 → resize → Lab转换 → 主体/背景分离 → 各自聚类 → 评分 → JSON/终端输出
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
from .visualize import save_result_preview, save_segmentation_visualization
from .html_visualize import HTMLVisualizer


DEFAULT_CONFIG = {
    # 预处理
    "max_size": 512,
    # 分割（3个核心参数，控制主体识别的激进程度）
    "chroma_threshold": 10.0,       # 色度阈值：越低→识别越激进，更多彩色区域被纳入主体
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
    "weight_variance": 0.10,
    "weight_center": 0.1,
    "weight_chroma": 0.3,
    "min_score_display": 0.05,
    # 视觉显著性参数
    "salience_a_weight": 1.3,          # 色度计算中 a* 的加权: chroma=sqrt(a_weight*a²+b²)
    "salience_chroma_logit": 2.0,      # 色度/亮度权重比值 logit, sigmoid后 w_chroma=w_L=0.5
}


class GraphColorPipeline:
    """
    主色提取管线。
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
            min_score_display=self.config["min_score_display"],
            salience_a_weight=self.config.get("salience_a_weight", 1.2),
            salience_chroma_logit=self.config.get("salience_chroma_logit", 0.0),
        )

    def warmup(self):
        """预加载分割模型，避免首张图片处理耗时过长。"""
        if hasattr(self.segmenter, "warmup"):
            self.segmenter.warmup()

    def process(self, image_path: str) -> ImageResult:
        """
        处理单张图片。

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

                bg_result = analyze_region(
                    "background", bg_pixels, bg_coords,
                    self.bg_clusterer, self.scorer, h, w
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

                fg_result = analyze_region(
                    "foreground", fg_pixels, fg_coords,
                    self.fg_clusterer, self.scorer, h, w
                )

                seg_result = _make_result(fg_mask, f"neural_{self.segmenter.model_name}")
                seg_result.initial_mask = neural_mask  # 保存首次提取的mask用于可视化

                return ImageResult(
                    image_path=str(path),
                    foreground=fg_result,
                    background=bg_result,
                    segment_method=seg_result.method,
                    seg_result=seg_result,
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
        fg_result = analyze_region(
            "foreground", fg_pixels, fg_coords,
            self.fg_clusterer, self.scorer, h, w
        )

        # 步骤6: 背景分析
        bg_result = analyze_region(
            "background", bg_pixels, bg_coords,
            self.bg_clusterer, self.scorer, h, w
        )

        return ImageResult(
            image_path=str(path),
            foreground=fg_result,
            background=bg_result,
            segment_method=seg_result.method,
            seg_result=seg_result,
        )

    def process_batch(self, image_paths: List[str],
                      verbose: bool = True,
                      workers: int = 1,
                      output_preview_dir: Optional[str] = None,
                      output_seg_visual_dir: Optional[str] = None,
                      output_html: Optional[str] = None) -> List[ImageResult]:
        """
        批量处理多张图片。

        Args:
            image_paths: 图片路径列表
            verbose: 是否打印进度
            workers: 并行工作进程数 (1=单线程, >1=多进程)
            output_preview_dir: 若有值，流式保存带主色色块的结果图
            output_seg_visual_dir: 若有值，流式保存主体识别可视化图
            output_html: 若有值，流式输出 HTML 可视化结果

        Returns:
            ImageResult列表
        """
        # Warmup: 预加载分割模型（不计入计时）
        self.warmup()

        if workers <= 1:
            return self._process_batch_sequential(
                image_paths, verbose, output_preview_dir, output_seg_visual_dir, output_html)

        return self._process_batch_parallel(
            image_paths, verbose, workers, output_preview_dir, output_seg_visual_dir, output_html)

    def _process_batch_sequential(self, image_paths: List[str],
                                   verbose: bool,
                                   preview_dir: Optional[str] = None,
                                   seg_visual_dir: Optional[str] = None,
                                   output_html: Optional[str] = None) -> List[ImageResult]:
        """单线程顺序处理（流式输出结果图）"""
        results = []
        total = len(image_paths)
        errors = 0
        t_start = time.perf_counter()

        if preview_dir:
            Path(preview_dir).mkdir(parents=True, exist_ok=True)
        if seg_visual_dir:
            Path(seg_visual_dir).mkdir(parents=True, exist_ok=True)

        used_names_preview: set[str] = set()
        used_names_seg: set[str] = set()

        # HTML 可视化器（流式写入）
        html_viz = HTMLVisualizer(output_html) if output_html else None

        for i, path in enumerate(image_paths):
            try:
                result = self.process(path)
                results.append(result)

                # 流式保存结果图
                if preview_dir:
                    preview_path = _preview_path_for(
                        result.image_path, preview_dir, used_names_preview)
                    save_result_preview(result.image_path, result, preview_path)

                # 流式保存主体可视化图
                if seg_visual_dir and result.seg_result is not None:
                    seg_path = _preview_path_for(
                        result.image_path, seg_visual_dir, used_names_seg)
                    save_segmentation_visualization(result.image_path, result.seg_result, seg_path)

                # 流式写入 HTML
                if html_viz:
                    html_viz.add_result(image_result_to_dict(result))

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

        # 关闭 HTML 文件
        if html_viz:
            html_viz.close()
            if verbose:
                print(f"  HTML 可视化已保存至: {output_html}")

        return results

    def _process_batch_parallel(self, image_paths: List[str],
                                 verbose: bool,
                                 workers: int,
                                 preview_dir: Optional[str] = None,
                                 seg_visual_dir: Optional[str] = None,
                                 output_html: Optional[str] = None) -> List[ImageResult]:
        """多进程并行处理（流式输出结果图）"""
        from concurrent.futures import ProcessPoolExecutor, as_completed
        import threading

        results = [None] * len(image_paths)
        errors = 0
        t_start = time.perf_counter()

        if preview_dir:
            Path(preview_dir).mkdir(parents=True, exist_ok=True)
        if seg_visual_dir:
            Path(seg_visual_dir).mkdir(parents=True, exist_ok=True)

        used_names_preview: set[str] = set()
        used_names_seg: set[str] = set()
        used_names_preview_lock = threading.Lock()
        used_names_seg_lock = threading.Lock()

        # HTML 可视化器（流式写入，多线程安全）
        html_viz = HTMLVisualizer(output_html) if output_html else None
        html_lock = threading.Lock() if html_viz else None

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

                    # 流式保存结果图
                    if preview_dir:
                        with used_names_preview_lock:
                            preview_path = _preview_path_for(
                                result.image_path, preview_dir, used_names_preview)
                        save_result_preview(result.image_path, result, preview_path)

                    # 流式保存主体可视化图
                    if seg_visual_dir and result.seg_result is not None:
                        with used_names_seg_lock:
                            seg_path = _preview_path_for(
                                result.image_path, seg_visual_dir, used_names_seg)
                        save_segmentation_visualization(result.image_path, result.seg_result, seg_path)

                    # 流式写入 HTML
                    if html_viz and html_lock:
                        with html_lock:
                            html_viz.add_result(image_result_to_dict(result))

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

        # 关闭 HTML 文件
        if html_viz:
            html_viz.close()
            if verbose:
                print(f"  HTML 可视化已保存至: {output_html}")

        return results

    @staticmethod
    def _process_single(image_path: str, config: dict) -> ImageResult:
        """多进程工作函数：创建独立管线处理单张图片"""
        p = GraphColorPipeline(config)
        p.warmup()
        return p.process(image_path)


def image_result_to_dict(result: ImageResult) -> dict:
    """将ImageResult转换为可JSON序列化的字典"""
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


def _preview_path_for(image_path: str, preview_dir: str,
                      used_names: set[str]) -> str:
    stem = Path(image_path).stem
    safe_stem = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in stem)
    candidate = f"{safe_stem}_graphcolor.png"
    i = 2
    while candidate.lower() in used_names:
        candidate = f"{safe_stem}_{i}_graphcolor.png"
        i += 1
    used_names.add(candidate.lower())
    return str(Path(preview_dir) / candidate)


def process_batch(image_paths: List[str],
                  config: Optional[dict] = None,
                  output_json: Optional[str] = None,
                  output_preview_dir: Optional[str] = None,
                  output_seg_visual_dir: Optional[str] = None,
                  output_html: Optional[str] = None,
                  verbose: bool = True,
                  workers: int = 1) -> List[dict]:
    """
    便捷的批量处理入口函数。

    Args:
        image_paths: 图片路径列表
        config: 配置覆盖字典
        output_json: 若有值，将结果写入此JSON文件
        output_preview_dir: 若有值，将保存带两个主色色块的结果图
        output_seg_visual_dir: 若有值，将保存主体识别可视化图
        output_html: 若有值，将流式输出 HTML 可视化结果
        verbose: 是否打印进度
        workers: 并行工作进程数 (1=单线程, >1=多进程)

    Returns:
        字典列表（可直接序列化为JSON）
    """
    pipeline = GraphColorPipeline(config)

    # 默认预览目录
    if output_preview_dir is None:
        output_preview_dir = str(Path("outputs/previews"))

    results = pipeline.process_batch(
        image_paths, verbose=verbose, workers=workers,
        output_preview_dir=output_preview_dir,
        output_seg_visual_dir=output_seg_visual_dir,
        output_html=output_html)
    dict_results = [image_result_to_dict(r) for r in results]

    if output_json:
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(dict_results, f, ensure_ascii=False, indent=2)
        if verbose:
            print(f"\n结果已保存至: {output_json}")

    return dict_results
