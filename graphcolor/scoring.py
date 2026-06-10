"""
聚类评分与主色提取模块。

评分体系基于 Lab 色彩空间的感知维度：
  1. 像素占比 —— 该颜色在区域中的覆盖率
  2. 亮度均匀性 —— 亮度越均匀分越高（代表颜色越纯）
  3. 空间中心距离 —— 越靠近图像中心的聚类权重越高
  4. 视觉显著性 —— 统一整合色度与亮度的感知显著性评分

视觉显著性算法（compute_visual_salience）：
  1. 加权色度: chroma = sqrt(a_weight * a² + b²)
  2. 归一化: L' = (L - mean_L) / std_L * 0.5, clip [-1,1]
            chroma' = (chroma - mean_chroma) / std_chroma * 0.5, clip [-1,1]
     目标: 均值=0, σ²=0.25 (即 σ=0.5)
  3. 权重: w_chroma = sigmoid(chroma_logit)
           w_L = 1 - w_chroma
  4. 评分: score_L = L'²  (极端亮度得分高)
           score_chroma = (chroma' + 1) / 2  (映射到 [0,1])
  5. 融合: score = w_chroma * score_chroma + w_L * (1 - score_chroma) * score_L
"""
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .segment import SegmentResult

from .cluster import ClusterResult, LabClusterer


@dataclass
class MainColor:
    """单个主色结果"""
    lab: np.ndarray          # (3,) 标准Lab值
    rgb: np.ndarray          # (3,) RGB值 (0-255)
    hex_color: str           # "#RRGGBB"
    score: float             # 综合评分 0~1
    proportion: float        # 像素占比 0~1
    cluster_id: int          # 聚类编号


@dataclass
class RegionResult:
    """单个区域（前景/背景）的主色分析结果"""
    region_name: str                    # "foreground" / "background"
    main_colors: List[MainColor]        # 按评分排序的主色列表
    dominant_color: MainColor           # 最高分主色


@dataclass
class ImageResult:
    """单张图片的完整分析结果"""
    image_path: str
    foreground: RegionResult
    background: RegionResult
    segment_method: str
    seg_result: "Optional[SegmentResult]" = None  # 用于可视化，不参与序列化


class ClusterScorer:
    """
    聚类评分器。
    每个聚类的主色被综合多个因子评分。
    """

    def __init__(self,
                 weight_count: float = 0.60,
                 weight_variance: float = 0.10,
                 weight_center: float = 0.15,
                 weight_chroma: float = 0.15,
                 min_score_display: float = 0.05,
                 salience_a_weight: float = 1.2,
                 salience_chroma_logit: float = 0.0):
        """
        Args:
            weight_count: 像素占比权重
            weight_variance: 亮度均匀性权重
            weight_center: 空间中心距离权重
            weight_chroma: 视觉显著性权重（占总评分的权重）
            min_score_display: 低于此分数的聚类不纳入主色输出
            salience_a_weight: 色度计算中 a* 通道的加权系数
                chroma = sqrt(a_weight * a² + b²)
            salience_chroma_logit: 色度/亮度权重比值的 logit 表示
                w_chroma = sigmoid(logit) = 1 / (1 + exp(-logit))
                w_L = 1 - w_chroma
                logit=0 → w_chroma=w_L=0.5
                logit>0 → 色度权重更大
                logit<0 → 亮度权重更大
        """
        self.weights = np.array([
            weight_count, weight_variance, weight_center, weight_chroma
        ])
        self.weights = self.weights / self.weights.sum()  # 归一化
        self.min_score_display = min_score_display
        self.salience_a_weight = salience_a_weight
        self.salience_chroma_logit = salience_chroma_logit

    @staticmethod
    def compute_brightness_variance(labels: np.ndarray,
                                    L_values: np.ndarray,
                                    n_clusters: int) -> np.ndarray:
        """
        向量化计算每个聚类的亮度方差。
        使用 np.bincount 避免 for 循环。
        """
        # 均值: sum(L)/count per cluster
        counts = np.bincount(labels, minlength=n_clusters).astype(np.float64)
        sums = np.bincount(labels, weights=L_values, minlength=n_clusters)
        means = sums / np.maximum(counts, 1)

        # 方差: sum((L - mean)²) / count
        # 使用: sum(L²) - 2*mean*sum(L) + count*mean² = sum(L²) - sum(L)²/count
        sq_sums = np.bincount(labels, weights=L_values ** 2, minlength=n_clusters)
        variances = sq_sums - sums ** 2 / np.maximum(counts, 1)
        variances = np.maximum(variances / np.maximum(counts, 1), 0)

        # 小聚类（<2 像素）方差设为 0
        variances[counts < 2] = 0

        # 反转：方差越小分越高
        mean_var = variances.mean()
        if mean_var > 0:
            scores = np.exp(-variances / (mean_var + 1e-8))
        else:
            scores = np.ones(n_clusters)
        return scores

    @staticmethod
    def compute_center_distances(labels: np.ndarray,
                                 coords: np.ndarray,
                                 n_clusters: int,
                                 img_h: int, img_w: int) -> np.ndarray:
        """
        向量化计算每个聚类到图像几何中心的平均距离。
        使用 np.bincount 避免 for 循环。
        """
        center_y, center_x = img_h / 2, img_w / 2
        max_dist = np.sqrt(center_y ** 2 + center_x ** 2)

        # 计算每个像素到中心的距离
        dists = np.sqrt(
            (coords[:, 0] - center_y) ** 2 +
            (coords[:, 1] - center_x) ** 2
        )

        # 每个聚类的平均距离: sum(dists) / count
        counts = np.bincount(labels, minlength=n_clusters).astype(np.float64)
        dist_sums = np.bincount(labels, weights=dists, minlength=n_clusters)
        distances = dist_sums / np.maximum(counts, 1)

        # 归一化为接近度 (越近越高)
        if max_dist > 0:
            proximity = 1.0 - distances / max_dist
        else:
            proximity = np.ones(n_clusters)
        return np.clip(proximity, 0, 1)

    def compute_visual_salience(self, centers_lab: np.ndarray,
                                region_mean_L: float,
                                region_std_L: float,
                                region_mean_chroma: float,
                                region_std_chroma: float) -> np.ndarray:
        """
        统一视觉显著性评分。

        算法步骤：
        1. 加权色度: chroma = sqrt(a_weight * a² + b²)
        2. 归一化: L' = (L - mean_L) / std_L * 0.5, clip [-1,1]
                  chroma' = (chroma - mean_chroma) / std_chroma * 0.5, clip [-1,1]
           目标: 均值=0, σ²=0.25 (即 σ=0.5)
        3. 权重: w_chroma = sigmoid(chroma_logit)
                 w_L = 1 - w_chroma
        4. 评分: score_L = L'²  (极端亮度得分高)
                 score_chroma = (chroma' + 1) / 2  (映射到 [0,1])
        5. 融合: score = w_chroma * score_chroma + w_L * (1 - score_chroma) * score_L
        """
        a = centers_lab[:, 1]
        b = centers_lab[:, 2]

        # --- 步骤1: 加权色度 ---
        chroma = np.sqrt(self.salience_a_weight * a ** 2 + b ** 2)

        # --- 步骤2: 归一化 (均值=0, σ=0.5, clip [-1,1]) ---
        safe_std_L = max(region_std_L, 1e-6)
        safe_std_chroma = max(region_std_chroma, 1e-6)

        L_norm = np.clip((centers_lab[:, 0] - region_mean_L) / safe_std_L * 0.5, -1.0, 1.0)
        chroma_norm = np.clip((chroma - region_mean_chroma) / safe_std_chroma * 0.5, -1.0, 1.0)

        # --- 步骤3: 权重 ---
        w_chroma = 1.0 / (1.0 + np.exp(-self.salience_chroma_logit))
        w_L = 1.0 - w_chroma

        # --- 步骤4: 评分 ---
        score_L = L_norm ** 2
        score_chroma = (chroma_norm + 1.0) / 2.0

        # --- 步骤5: 融合 ---
        final_salience = (
            w_chroma * score_chroma +
            w_L * (1.0 - score_chroma) * score_L
        )

        return np.clip(final_salience, 0, 1)

    def score(self, cluster_result: ClusterResult,
              img_h: int, img_w: int,
              pixel_coords: Optional[np.ndarray] = None,
              L_values: Optional[np.ndarray] = None,
              labels: Optional[np.ndarray] = None,
              region_mean_L: Optional[float] = None,
              region_std_L: Optional[float] = None,
              region_mean_chroma: Optional[float] = None,
              region_std_chroma: Optional[float] = None) -> List[MainColor]:
        """
        对聚类结果评分并输出主色列表。

        Args:
            cluster_result: 聚类结果
            img_h: 图像高度
            img_w: 图像宽度
            pixel_coords: (N, 2) 每个像素的 [y, x] 坐标
            L_values: (N,) 每个像素的 L* 值
            labels: (N,) 聚类标签 (若未提供则使用cluster_result中的)
            region_mean_L: 当前区域平均亮度（归一化用）
            region_std_L: 当前区域亮度标准差（归一化用）
            region_mean_chroma: 当前区域平均色度（归一化用）
            region_std_chroma: 当前区域色度标准差（归一化用）

        Returns:
            按综合评分降序排列的主色列表
        """
        n_clusters = len(cluster_result.centers_lab)
        if n_clusters == 0:
            return []

        centers_lab = cluster_result.centers_lab
        proportions = cluster_result.proportions

        # --- 计算各因子分数 ---
        # 1. 像素占比 (直接使用归一化的proportion)
        count_scores = proportions

        # 2. 亮度方差
        if L_values is not None and labels is not None:
            variance_scores = self.compute_brightness_variance(
                labels, L_values, n_clusters)
        else:
            variance_scores = np.ones(n_clusters) * 0.5

        # 3. 空间中心距离
        if pixel_coords is not None and labels is not None:
            center_scores = self.compute_center_distances(
                labels, pixel_coords, n_clusters, img_h, img_w)
        else:
            center_scores = np.ones(n_clusters) * 0.5

        # 4. 视觉显著性（统一整合色度+亮度的感知评分）
        salience_scores = self.compute_visual_salience(
            centers_lab,
            region_mean_L=region_mean_L,
            region_std_L=region_std_L,
            region_mean_chroma=region_mean_chroma,
            region_std_chroma=region_std_chroma
        )

        # --- 综合评分 ---
        all_scores = np.stack([
            count_scores, variance_scores,
            center_scores, salience_scores
        ], axis=1)

        final_scores = all_scores @ self.weights

        # --- 排序输出 ---
        sort_idx = np.argsort(-final_scores)
        valid_mask = final_scores[sort_idx] >= self.min_score_display
        sort_idx = sort_idx[valid_mask]

        if len(sort_idx) == 0:
            return []

        # 批量 Lab -> RGB 转换
        all_labs = centers_lab[sort_idx]
        all_rgbs = self._lab_to_rgb_batch(all_labs)

        # 批量生成 hex
        rgb_int = np.clip(np.round(all_rgbs), 0, 255).astype(np.int32)
        hex_colors = [
            "#{:02X}{:02X}{:02X}".format(r, g, b)
            for r, g, b in rgb_int
        ]

        return [
            MainColor(
                lab=all_labs[i],
                rgb=all_rgbs[i],
                hex_color=hex_colors[i],
                score=float(final_scores[sort_idx[i]]),
                proportion=float(proportions[sort_idx[i]]),
                cluster_id=int(sort_idx[i]),
            )
            for i in range(len(sort_idx))
        ]

    @staticmethod
    def _lab_to_rgb_batch(labs: np.ndarray) -> np.ndarray:
        """批量 (k, 3) Lab -> RGB (k, 3), 0-255"""
        import cv2
        n = len(labs)
        if n == 0:
            return np.zeros((0, 3), dtype=np.float32)
        lab_img = labs.reshape(1, n, 3).astype(np.float32)
        lab_img[:, :, 0] = lab_img[:, :, 0] / 100.0 * 255.0
        lab_img[:, :, 1] = lab_img[:, :, 1] + 128.0
        lab_img[:, :, 2] = lab_img[:, :, 2] + 128.0
        lab_img = np.clip(lab_img, 0, 255).astype(np.uint8)
        bgr = cv2.cvtColor(lab_img, cv2.COLOR_Lab2BGR)
        # (1, k, 3) BGR -> (k, 3) RGB
        return bgr[0, :, ::-1].astype(np.float32)


def analyze_region(region_name: str,
                   lab_pixels: np.ndarray,
                   pixel_coords: np.ndarray,
                   clusterer: LabClusterer,
                   scorer: ClusterScorer,
                   img_h: int, img_w: int) -> RegionResult:
    """
    分析单个区域（前景或背景）的主色。

    Args:
        region_name: 区域名称
        lab_pixels: (N, 3) Lab像素值
        pixel_coords: (N, 2) 像素坐标 [y, x]
        clusterer: 聚类器
        scorer: 评分器
        img_h: 图像高度
        img_w: 图像宽度

    Returns:
        RegionResult
    """
    if len(lab_pixels) == 0:
        # 空区域
        empty_dominant = MainColor(
            lab=np.zeros(3), rgb=np.zeros(3),
            hex_color="#000000", score=0.0,
            proportion=0.0, cluster_id=-1
        )
        return RegionResult(
            region_name=region_name,
            main_colors=[],
            dominant_color=empty_dominant
        )

    # 聚类
    result = clusterer.fit(lab_pixels)

    # 计算区域统计量（用于归一化）
    L_values = lab_pixels[:, 0]
    a_values = lab_pixels[:, 1]
    b_values = lab_pixels[:, 2]
    chroma_all = np.sqrt(a_values ** 2 + b_values ** 2)
    region_mean_L = float(L_values.mean())
    region_std_L = float(L_values.std())
    region_mean_chroma = float(chroma_all.mean())
    region_std_chroma = float(chroma_all.std())

    # 评分
    main_colors = scorer.score(
        result, img_h, img_w,
        pixel_coords=pixel_coords,
        L_values=L_values,
        labels=result.labels,
        region_mean_L=region_mean_L,
        region_std_L=region_std_L,
        region_mean_chroma=region_mean_chroma,
        region_std_chroma=region_std_chroma,
    )

    dominant = main_colors[0] if main_colors else MainColor(
        lab=np.zeros(3), rgb=np.zeros(3),
        hex_color="#000000", score=0.0,
        proportion=0.0, cluster_id=-1
    )

    return RegionResult(
        region_name=region_name,
        main_colors=main_colors[:5],  # 最多输出5个主色
        dominant_color=dominant
    )
