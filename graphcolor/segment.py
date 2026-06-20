"""
主体/背景分离模块。

提供两种分割器:
  - ForegroundSegmenter: 基于 OpenCV GrabCut 的多策略融合方案(轻量、无额外依赖,
                        适合 CPU / 无深度学习环境)
  - NeuralSegmenter:    基于 rembg(U2-Net / IS-Net)的深度学习显著性检测方案
                        (更精准,适合动漫/插画等复杂背景)

此外还提供一些公共工具:
  - SegmentResult          统一封装两个 mask + method + foreground_ratio
  - _clean_foreground      形态学后处理 + 连通域过滤(去小斑点、填洞)
  - _make_result / _result 把 bool mask 装成 SegmentResult
  - _center_weight         距离图像中心的归一化权重,给 GrabCut 当作先验

NeuralSegmenter.refine_mask() 是 pipeline 调用的关键:在拿到背景主色后,
用它二次精修初始 mask, 解决"环状主体内部空洞被误识别为前景"的问题。
"""
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass
class SegmentResult:
    """分割结果

    Attributes:
        foreground_mask:  (H, W) bool, 精修后的前景 mask(主要被后续 pipeline 使用的字段)
        background_mask:  (H, W) bool, 前景的反面
        method:           使用的方法名(grabcut_seeded / neural_isnet-general-use / ...),
                          用于调试输出与可视化标注
        foreground_ratio: 前景占总像素的比例(0~1),方便异常检测
        initial_mask:     首次提取的 mask(未经 refine),仅 visualize 用来对比展示
    """
    foreground_mask: np.ndarray      # bool, shape (H, W) - 精修后的前景mask
    background_mask: np.ndarray      # bool, shape (H, W)
    method: str                      # 使用的方法名
    foreground_ratio: float = 0.0
    initial_mask: np.ndarray = None  # bool, shape (H, W) - 首次提取的mask（用于可视化对比）


# ────────────────────────────────────────────
# 公共工具函数（两个分割器共享）
# ────────────────────────────────────────────

def _clean_foreground(mask: np.ndarray, kernel_close: int = 5,
                      kernel_open: int = 3) -> np.ndarray:
    """形态学后处理：闭运算 → 开运算 → 连通域过滤"""
    mask = _morph(mask, cv2.MORPH_CLOSE, kernel_close, 1)
    mask = _morph(mask, cv2.MORPH_OPEN, kernel_open, 1)
    mask_u8 = mask.astype(np.uint8)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, 8)
    if n_labels <= 1:
        return mask.astype(bool)

    h, w = mask.shape[:2]
    total = h * w
    keep = np.zeros(n_labels, dtype=bool)
    min_keep = max(24, int(total * 0.002))
    center_weight = _center_weight(h, w)
    for label in range(1, n_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        component = labels == label
        center_score = float(center_weight[component].mean()) if area else 0.0
        if area >= min_keep or (area >= min_keep // 2 and center_score > 0.45):
            keep[label] = True
    return keep[labels]


def _morph(mask: np.ndarray, op: int, kernel_size: int,
           iterations: int) -> np.ndarray:
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
    )
    return cv2.morphologyEx(
        mask.astype(np.uint8), op, kernel, iterations=iterations
    ).astype(bool)


def _make_result(foreground: np.ndarray, method: str) -> SegmentResult:
    foreground = foreground.astype(bool)
    return SegmentResult(
        foreground_mask=foreground,
        background_mask=~foreground,
        method=method,
        foreground_ratio=float(foreground.mean())
    )


def _center_weight(h: int, w: int) -> np.ndarray:
    yy, xx = np.ogrid[:h, :w]
    center_y, center_x = h / 2.0, w / 2.0
    max_dist = np.sqrt(center_y ** 2 + center_x ** 2)
    dist = np.sqrt((yy - center_y) ** 2 + (xx - center_x) ** 2)
    return np.clip(1.0 - dist / max(max_dist, 1e-6), 0.0, 1.0)


class ForegroundSegmenter:
    """
    主体/背景分离器(传统方案,基于多线索 + GrabCut)。

    3个核心参数(控制主体识别的激进程度):
      - chroma_threshold: 色度阈值,越低→更多彩色区域被纳入主体
      - min_contour_area_ratio: 最小轮廓面积比,越低→保留更小的轮廓细节
      - foreground_score_threshold: 前景得分阈值,越低→主体识别越激进

    算法思路(segment):
      1) 优先使用 alpha 通道(若 PNG 自带透明)
      2) 在 Lab 空间下融合 4 条线索:chroma / 边缘 / 亮度中心 / 中心距离,
         再扣去"和图像边缘颜色相近"的区域(避免把白墙等大片背景纳入主体)
      3) 以"强前景 + 弱前景 + 弱背景"作为 GrabCut 的 mask 种子
      4) 失败兜底:基于 score 的阈值法 / 椭圆中心
    """

    # 内部常量参数（次要，通常不需要调整）
    _EDGE_WEIGHT = 0.6
    _GRABCUT_ITERATIONS = 2
    _BORDER_RATIO = 0.04

    def __init__(self, chroma_threshold: float = 12.0,
                 min_contour_area_ratio: float = 0.03,
                 foreground_score_threshold: float = 0.55):
        """
        Args:
            chroma_threshold: Lab 色度阈值。越低→更多彩色区域被纳入主体候选。
            min_contour_area_ratio: 轮廓保留的最小面积占比。越低→保留更小的轮廓细节。
            foreground_score_threshold: 前景得分阈值。越低→主体识别越激进。
        """
        self.chroma_threshold = chroma_threshold
        self.min_contour_area_ratio = min_contour_area_ratio
        self.foreground_score_threshold = foreground_score_threshold

    def segment(self, lab_img: np.ndarray,
                bgr_img: Optional[np.ndarray] = None,
                alpha_mask: Optional[np.ndarray] = None) -> SegmentResult:
        """
        综合多策略分离主体与背景。

        Args:
            lab_img: Lab 图像 (H, W, 3), float32, L[0,100], a/b[-128,127]。
            bgr_img: 可选 BGR 图像，提供后会启用 GrabCut。
            alpha_mask: 可选 alpha 通道；透明背景图片优先使用 alpha 分割。
        """
        h, w = lab_img.shape[:2]
        total_pixels = h * w

        if alpha_mask is not None and alpha_mask.shape[:2] == (h, w):
            alpha_foreground = alpha_mask > 16
            ratio = alpha_foreground.mean()
            if 0.005 <= ratio <= 0.995:
                foreground = _clean_foreground(alpha_foreground)
                return _make_result(foreground, "alpha")

        cues = self._build_cues(lab_img)
        foreground = self._grabcut(lab_img, bgr_img, cues)
        if foreground is not None:
            return _make_result(foreground, "grabcut_seeded")

        # Fast fallback for cases where GrabCut cannot be initialized robustly.
        vote = cues["score"]
        fallback = vote >= 0.60
        if fallback.mean() < 0.005:
            fallback = cues["strong_foreground"]
        if fallback.mean() < 0.005:
            # Last resort: use a centered ellipse instead of returning an empty foreground.
            yy, xx = np.ogrid[:h, :w]
            cy, cx = h / 2, w / 2
            ry, rx = max(1.0, h * 0.38), max(1.0, w * 0.38)
            fallback = ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1.0

        foreground = _clean_foreground(fallback)
        return _make_result(foreground, "lab_edge_fallback")

    def _build_cues(self, lab_img: np.ndarray) -> dict:
        h, w = lab_img.shape[:2]
        total_pixels = h * w
        min_area = max(16, int(total_pixels * self.min_contour_area_ratio))

        L = lab_img[:, :, 0]
        a = lab_img[:, :, 1]
        b = lab_img[:, :, 2]
        chroma = np.sqrt(a ** 2 + b ** 2)

        # Adaptive chroma keeps pale backgrounds from becoming foreground wholesale.
        chroma_cut = max(
            self.chroma_threshold,
            float(np.median(chroma) + 0.45 * np.std(chroma))
        )
        mask_chroma = chroma > chroma_cut
        mask_chroma = self._morph(mask_chroma, cv2.MORPH_OPEN, 5, 1)

        border_mask = self._border_mask(h, w)
        border_lab = lab_img[border_mask]
        border_median = np.median(border_lab, axis=0) if len(border_lab) else np.zeros(3)
        delta_border = np.sqrt(
            ((lab_img[:, :, 0] - border_median[0]) * 0.45) ** 2 +
            ((lab_img[:, :, 1] - border_median[1]) * 1.10) ** 2 +
            ((lab_img[:, :, 2] - border_median[2]) * 1.10) ** 2
        )
        border_delta = delta_border[border_mask]
        bg_cut = max(9.0, float(np.percentile(border_delta, 85) + 4.0))
        bg_like = delta_border <= bg_cut

        L_uint8 = np.clip(L / 100.0 * 255.0, 0, 255).astype(np.uint8)
        edges = cv2.Canny(L_uint8, 30, 100)
        edge_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        edges = cv2.dilate(edges, edge_kernel, iterations=1)
        closing = cv2.morphologyEx(
            edges, cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
            iterations=2
        )
        contours, _ = cv2.findContours(
            closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        mask_edge = np.zeros((h, w), dtype=np.uint8)
        valid_contours = [c for c in contours if cv2.contourArea(c) >= min_area]
        if valid_contours:
            # Keep several large contours; multiple characters or props are common.
            valid_contours = sorted(valid_contours, key=cv2.contourArea, reverse=True)[:5]
            cv2.drawContours(mask_edge, valid_contours, -1, 1, thickness=cv2.FILLED)
        mask_edge = mask_edge.astype(bool)

        center_weight = self._center_weight(h, w)
        center_mask = center_weight >= 0.25

        center_l = np.median(L[h // 4: max(h // 4 + 1, 3 * h // 4),
                               w // 4: max(w // 4 + 1, 3 * w // 4)])
        border_l = np.median(L[border_mask]) if border_mask.any() else np.median(L)
        mask_luminance = np.zeros((h, w), dtype=bool)
        if abs(float(center_l - border_l)) > 10.0 or L.std() > 22.0:
            _, otsu = cv2.threshold(
                L_uint8, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            bright_side = otsu.astype(bool)
            if center_l >= border_l:
                mask_luminance = bright_side
            else:
                mask_luminance = ~bright_side
            mask_luminance = self._morph(mask_luminance, cv2.MORPH_OPEN, 5, 1)

        score = np.zeros((h, w), dtype=np.float32)
        score += mask_chroma.astype(np.float32) * 1.00
        score += mask_edge.astype(np.float32) * self._EDGE_WEIGHT
        score += mask_luminance.astype(np.float32) * 0.75
        score += center_weight.astype(np.float32) * 0.30
        score -= bg_like.astype(np.float32) * 0.65
        score[border_mask] -= 0.50

        strong_foreground = (score >= self.foreground_score_threshold) & center_mask & ~border_mask
        if strong_foreground.sum() < max(8, int(total_pixels * 0.003)):
            threshold = np.percentile(score[center_mask], 88) if center_mask.any() else np.percentile(score, 90)
            strong_foreground = (score >= threshold) & center_mask & ~border_mask

        probable_foreground = (score >= 0.25) & ~border_mask
        probable_background = bg_like | border_mask

        return {
            "score": score,
            "strong_foreground": strong_foreground,
            "probable_foreground": probable_foreground,
            "probable_background": probable_background,
            "border_mask": border_mask,
        }

    def _grabcut(self, lab_img: np.ndarray, bgr_img: Optional[np.ndarray],
                 cues: dict) -> Optional[np.ndarray]:
        if bgr_img is None:
            return None
        h, w = lab_img.shape[:2]
        if h < 8 or w < 8 or bgr_img.shape[:2] != (h, w):
            return None

        strong_fg = cues["strong_foreground"]
        probable_fg = cues["probable_foreground"]
        probable_bg = cues["probable_background"]

        total = h * w
        if strong_fg.sum() < max(8, int(total * 0.002)):
            return None
        if probable_bg.sum() < max(8, int(total * 0.002)):
            return None

        mask = np.full((h, w), cv2.GC_PR_BGD, dtype=np.uint8)
        mask[probable_bg] = cv2.GC_BGD
        mask[probable_fg] = cv2.GC_PR_FGD
        mask[strong_fg] = cv2.GC_FGD

        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        try:
            cv2.grabCut(
                bgr_img, mask, None, bgd_model, fgd_model,
                self._GRABCUT_ITERATIONS, cv2.GC_INIT_WITH_MASK
            )
        except cv2.error:
            return None

        foreground = (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)
        foreground = _clean_foreground(foreground)
        ratio = foreground.mean()
        if ratio < 0.005 or ratio > 0.985:
            return None
        return foreground

    def _border_mask(self, h: int, w: int) -> np.ndarray:
        border = max(2, int(round(min(h, w) * self._BORDER_RATIO)))
        mask = np.zeros((h, w), dtype=bool)
        mask[:border, :] = True
        mask[-border:, :] = True
        mask[:, :border] = True
        mask[:, -border:] = True
        return mask


class NeuralSegmenter:
    """
    基于 rembg(U2-Net / IS-Net)的深度学习主体检测分割器。

    相比传统 GrabCut 方案,深度学习显著性检测对主体识别更精准,
    尤其适合动漫/插画等复杂背景场景。

    提供了三段式 API 供 pipeline 调用:
      - extract_mask()  拉一次 rembg 得到初始 mask
      - refine_mask()    用背景主色 + GrabCut 二次精修,解决"环状主体内部空洞"
      - segment()       一站式便捷接口(内部自动完成精修)

    也可以单独传一个 alpha_mask,优先级最高的还是 alpha 通道,
    这对自带透明背景的 PNG(线稿/立绘)非常友好。
    """

    MODELS = {
        "isnet-general-use": "通用场景",
        "u2net": "经典显著性检测（较慢）",
        "u2netp": "轻量版（较快）",
        "u2net_human_seg": "人像分割",
        "silueta": "极轻量轮廓分割",
    }

    def __init__(self, model_name: str = "isnet-general-use"):
        """
        Args:
            model_name: rembg 模型名称。
                推荐 "isnet-general-use"（通用）或 "u2netp"（轻量）。
                动漫图片可考虑 "isnet-anime"（如果 rembg 版本支持）。
        """
        self.model_name = model_name
        self._session = None

    def _get_session(self):
        """懒加载 rembg session（首次调用时才加载模型）"""
        if self._session is None:
            from rembg import new_session
            self._session = new_session(self.model_name)
        return self._session

    def warmup(self):
        """预加载模型到内存，避免首张图片处理耗时过长。"""
        if self._session is None:
            self._get_session()

    def extract_mask(self, bgr_img: np.ndarray,
                     alpha_mask: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """
        使用 rembg 提取初始主体 mask，供 pipeline 做进一步精修。

        Args:
            bgr_img: BGR 图像 (H, W, 3), uint8
            alpha_mask: 可选 alpha 通道

        Returns:
            bool mask (H, W) 或 None（提取失败时）
        """
        h, w = bgr_img.shape[:2]

        # 优先使用 alpha 通道
        if alpha_mask is not None and alpha_mask.shape[:2] == (h, w):
            alpha_foreground = alpha_mask > 16
            ratio = alpha_foreground.mean()
            if 0.005 <= ratio <= 0.995:
                return _clean_foreground(alpha_foreground)

        try:
            from PIL import Image
            from rembg import remove

            session = self._get_session()
            rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            result_pil = remove(pil_img, session=session)
            alpha = np.array(result_pil)[:, :, 3]
            neural_mask = alpha > 16

            ratio = neural_mask.mean()
            if ratio < 0.005 or ratio > 0.995:
                return None
            return _clean_foreground(neural_mask)
        except Exception:
            return None

    def segment(self, lab_img: np.ndarray,
                bgr_img: Optional[np.ndarray] = None,
                alpha_mask: Optional[np.ndarray] = None) -> SegmentResult:
        """
        使用 rembg 深度学习模型进行主体/背景分离。

        注：完整流程（背景主色提取 → GrabCut 空洞精修）应由 pipeline 控制。
        此方法保留为便捷接口，内部自动完成精修。

        Args:
            lab_img: Lab 图像（用于回退方案）
            bgr_img: BGR 图像 (H, W, 3), uint8
            alpha_mask: 可选 alpha 通道
        """
        h, w = lab_img.shape[:2]

        if bgr_img is None:
            return self._fallback_segment(lab_img, h, w)

        # 提取初始 mask
        neural_mask = self.extract_mask(bgr_img, alpha_mask)
        if neural_mask is None:
            return self._fallback_segment(lab_img, h, w)

        # 用 GrabCut 进行二次精修
        foreground = self.refine_mask(bgr_img, neural_mask)
        if foreground is None:
            foreground = neural_mask

        ratio = foreground.mean()
        if ratio < 0.005:
            return self._fallback_segment(lab_img, h, w)

        return _make_result(foreground, f"neural_{self.model_name}")

    def refine_mask(self, bgr_img: np.ndarray,
                    neural_mask: np.ndarray,
                    bg_colors: np.ndarray,
                    hole_distance_threshold: float = 15.0,
                    hole_min_area_ratio: float = 0.002,
                    grabcut_iterations: int = 2) -> Optional[np.ndarray]:
        """
        使用 GrabCut 对初始 mask 进行精修，修复环状/中空区域的误识别。

        空洞检测使用正式的背景主色聚类结果，而非中位色。

        Args:
            bgr_img: BGR 图像
            neural_mask: 初始主体 mask
            bg_colors: 背景主色列表 (K, 3) BGR 值
            hole_distance_threshold: 空洞距离阈值（前景像素到最近背景主色的BGR欧氏距离上限）
            hole_min_area_ratio: 空洞最小面积比
            grabcut_iterations: GrabCut 迭代次数

        Returns:
            精修后的 mask 或 None
        """
        h, w = bgr_img.shape[:2]
        if h < 8 or w < 8 or bgr_img.shape[:2] != neural_mask.shape[:2]:
            return None

        if len(bg_colors) == 0:
            return _clean_foreground(neural_mask)

        # 前景区域：找与背景主色高度相近的区域
        fg_mask = neural_mask
        fg_pixels_bgr = bgr_img[fg_mask].astype(np.float32)
        if len(fg_pixels_bgr) < 16:
            return _clean_foreground(neural_mask)

        # 计算前景像素到最近的背景主色的距离
        # bg_colors: (K, 3), fg_pixels_bgr: (N, 3)
        # 使用广播计算最小距离: (N, K, 3) -> (N, K) -> (N,)
        diff = fg_pixels_bgr[:, np.newaxis, :] - bg_colors[np.newaxis, :, :]  # (N, K, 3)
        dists = np.sqrt(np.sum(diff ** 2, axis=2))  # (N, K)
        min_dists = np.min(dists, axis=1)  # (N,) - 每个像素到最近背景主色的距离

        hole_mask_in_fg = min_dists <= hole_distance_threshold

        # 映射回原图
        full_hole_mask = np.zeros((h, w), dtype=bool)
        full_hole_mask[fg_mask] = hole_mask_in_fg
        full_hole_mask = NeuralSegmenter._filter_internal_holes(full_hole_mask)

        if full_hole_mask.sum() == 0:
            return _clean_foreground(neural_mask)

        refined_mask = neural_mask & ~full_hole_mask

        gc_mask = np.full((h, w), cv2.GC_PR_BGD, dtype=np.uint8)
        gc_mask[full_hole_mask] = cv2.GC_BGD
        gc_mask[refined_mask] = cv2.GC_PR_FGD

        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            neural_mask.astype(np.uint8), 8)
        if n_labels > 1:
            border_labels = set()
            border_labels.update(np.unique(labels[0, :]))
            border_labels.update(np.unique(labels[-1, :]))
            border_labels.update(np.unique(labels[:, 0]))
            border_labels.update(np.unique(labels[:, -1]))

            min_area = max(16, int(h * w * hole_min_area_ratio))
            for label in range(1, n_labels):
                if label in border_labels:
                    continue
                area = stats[label, cv2.CC_STAT_AREA]
                if area < min_area:
                    continue
                component = labels == label
                cy, cx = np.where(component)
                cy, cx = int(cy.mean()), int(cx.mean())
                ry, rx = int(np.sqrt(area) * 0.45), int(np.sqrt(area) * 0.45)
                yy, xx = np.ogrid[:h, :w]
                center_region = ((yy - cy) / max(ry, 1)) ** 2 + ((xx - cx) / max(rx, 1)) ** 2 <= 1.0
                gc_mask[neural_mask & center_region] = cv2.GC_FGD

        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        try:
            cv2.grabCut(bgr_img, gc_mask, None, bgd_model, fgd_model,
                        grabcut_iterations, cv2.GC_INIT_WITH_MASK)
        except cv2.error:
            return _clean_foreground(refined_mask)

        foreground = (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD)
        foreground = foreground & neural_mask
        return _clean_foreground(foreground)

    @staticmethod
    def _filter_internal_holes(hole_candidates: np.ndarray) -> np.ndarray:
        """
        过滤空洞候选区域：只保留真正被前景包围的连通区域，
        排除与图像边界相连的区域（那些是真正的背景而非空洞）。
        """
        h, w = hole_candidates.shape[:2]
        mask_u8 = hole_candidates.astype(np.uint8)
        n_labels, labels, _, _ = cv2.connectedComponentsWithStats(mask_u8, 8)

        if n_labels <= 1:
            return np.zeros((h, w), dtype=bool)

        result = np.zeros((h, w), dtype=bool)

        # 标记所有与图像四边接触的连通区域
        border_labels = set()
        border_labels.update(np.unique(labels[0, :]))      # 上边
        border_labels.update(np.unique(labels[-1, :]))     # 下边
        border_labels.update(np.unique(labels[:, 0]))      # 左边
        border_labels.update(np.unique(labels[:, -1]))     # 右边

        for label in range(1, n_labels):
            if label not in border_labels:
                # 不与边界接触 → 真正的前景内部空洞
                result[labels == label] = True

        return result

    def _clean_foreground(self, mask: np.ndarray, h: int, w: int) -> np.ndarray:
        """形态学后处理"""
        mask = self._morph(mask, cv2.MORPH_CLOSE, 3, 1)
        mask = self._morph(mask, cv2.MORPH_OPEN, 3, 1)
        return mask.astype(bool)

    @staticmethod
    def _morph(mask: np.ndarray, op: int, kernel_size: int,
               iterations: int) -> np.ndarray:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        )
        return cv2.morphologyEx(
            mask.astype(np.uint8), op, kernel, iterations=iterations
        ).astype(bool)

    def _fallback_segment(self, lab_img: np.ndarray, h: int, w: int) -> SegmentResult:
        """深度学习失败时的椭圆中心兜底方案"""
        yy, xx = np.ogrid[:h, :w]
        cy, cx = h / 2, w / 2
        ry, rx = max(1.0, h * 0.38), max(1.0, w * 0.38)
        fallback = ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1.0
        return self._result(fallback, "neural_fallback")

    @staticmethod
    def _result(foreground: np.ndarray, method: str) -> SegmentResult:
        foreground = foreground.astype(bool)
        background = ~foreground
        return SegmentResult(
            foreground_mask=foreground,
            background_mask=background,
            method=method,
            foreground_ratio=float(foreground.mean())
        )
