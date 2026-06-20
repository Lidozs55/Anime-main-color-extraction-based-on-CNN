"""
聚类评分与主色提取模块。

评分体系基于 Lab 色彩空间的感知维度：
  1. 像素占比 —— 该颜色在区域中的覆盖率
  2. 亮度均匀性 —— 亮度越均匀分越高（代表颜色越纯）
  3. 空间中心距离 —— 越靠近图像中心的聚类权重越高
  4. 视觉显著性 —— 统一整合色度与亮度的感知显著性评分
  5. 肤色惩罚 —— 对人像图片中与肤色接近的颜色施加微弱折扣

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

肤色惩罚算法（compute_skin_similarity，仅 fixed 模式）：
  - 固定的 Lab 肤色参考 (ref, sigma)
  - 相似度 s = exp(-0.5 * Mahalanobis²)
  - 乘性折扣: final = final * (1 - s * skin_penalty_max)
  - skin_penalty_max 控制上限，默认 0.10
"""
import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .segment import SegmentResult

from .cluster import ClusterResult, LabClusterer


# ─── 固定肤色参考 (Lab 空间) ────────────────────────────────────────────
DEFAULT_SKIN_REF_LAB = np.array([74.0, 18.0, 24.0], dtype=np.float64)
DEFAULT_SKIN_SIGMA = np.array([16.0, 8.0, 10.0], dtype=np.float64)


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
    # 肤色惩罚快照：{'fg': dict, 'bg': dict}，由 pipeline 填充
    skin_info: Optional[dict] = None


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
                 weight_bilinear: float = 0.0,
                 min_score_display: float = 0.05,
                 salience_a_weight: float = 1.2,
                 salience_chroma_logit: float = 0.0,
                 skin_penalty_max: float = 0.10,
                 skin_ref_lab: Optional[np.ndarray] = None,
                 skin_sigma: Optional[np.ndarray] = None):
        """
        Args:
            weight_count: 像素占比权重 (count 线性项 a)
            weight_variance: 亮度均匀性权重
            weight_center: 空间中心距离权重
            weight_chroma: 视觉显著性权重 (salience 线性项 b)
            weight_bilinear: count × salience 乘性项系数 c。
                设为 0 时退化为纯加算 (旧行为)。
                >0 时: 高占比且高显著的颜色会得到"双高加成";
                       极小占比但高显著的颜色不会被加成 (避免小噪点上位)。
            min_score_display: 低于此分数的聚类不纳入主色输出
            salience_a_weight: 色度计算中 a* 通道的加权系数
                chroma = sqrt(a_weight * a² + b²)
            salience_chroma_logit: 色度/亮度权重比值的 logit 表示
                w_chroma = sigmoid(logit) = 1 / (1 + exp(-logit))
                w_L = 1 - w_chroma
                logit=0 → w_chroma=w_L=0.5
                logit>0 → 色度权重更大
                logit<0 → 亮度权重更大
            skin_penalty_max: 肤色惩罚上限 (0~1)。
                对"完美肤色匹配"的聚类乘以 (1 - skin_penalty_max) 的折扣。
                0 表示关闭肤色惩罚。
            skin_ref_lab: 固定肤色参考 (Lab, 3,)。None 时使用 DEFAULT_SKIN_REF_LAB。
            skin_sigma: 固定肤色标准差 (Lab, 3,)。None 时使用 DEFAULT_SKIN_SIGMA。
        """
        self.weights = np.array([
            weight_count, weight_variance, weight_center, weight_chroma
        ])
        self.weights = self.weights / self.weights.sum()  # 归一化 (仅线性权重)
        self.weight_bilinear = float(max(0.0, weight_bilinear))  # c, 单独作用,不参与归一化
        self.min_score_display = min_score_display
        self.salience_a_weight = salience_a_weight
        self.salience_chroma_logit = salience_chroma_logit

        # 肤色惩罚 (作为 score() 调用的默认配置，可被调用参数覆盖)
        self.skin_penalty_max = float(max(0.0, skin_penalty_max))
        self.skin_ref_lab = (
            np.asarray(skin_ref_lab, dtype=np.float64).copy()
            if skin_ref_lab is not None
            else DEFAULT_SKIN_REF_LAB.copy()
        )
        self.skin_sigma = (
            np.asarray(skin_sigma, dtype=np.float64).copy()
            if skin_sigma is not None
            else DEFAULT_SKIN_SIGMA.copy()
        )

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

    @staticmethod
    def compute_skin_similarity(centers_lab: np.ndarray,
                                skin_ref_lab: np.ndarray,
                                skin_sigma: np.ndarray) -> np.ndarray:
        """
        计算聚类中心与肤色的相似度，返回 [0, 1] 区间的向量。
        使用 Lab 空间高斯: s = exp(-0.5 * Σ((c-ref)/sigma)²)
        """
        diff = centers_lab - skin_ref_lab
        inv_sigma2 = 1.0 / (skin_sigma ** 2 + 1e-6)
        maha_sq = np.sum(diff * diff * inv_sigma2, axis=1)
        return np.exp(-0.5 * maha_sq)

    def score(self, cluster_result: ClusterResult,
              img_h: int, img_w: int,
              pixel_coords: Optional[np.ndarray] = None,
              L_values: Optional[np.ndarray] = None,
              labels: Optional[np.ndarray] = None,
              region_mean_L: Optional[float] = None,
              region_std_L: Optional[float] = None,
              region_mean_chroma: Optional[float] = None,
              region_std_chroma: Optional[float] = None,
              skin_penalty_max: Optional[float] = None,
              skin_ref_lab: Optional[np.ndarray] = None,
              skin_sigma: Optional[np.ndarray] = None) -> List[MainColor]:
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
            skin_penalty_max: 覆盖实例默认的肤色惩罚上限；None 则用 self.skin_penalty_max
            skin_ref_lab: 覆盖实例默认的肤色参考；None 则用 self.skin_ref_lab
            skin_sigma: 覆盖实例默认的肤色标准差；None 则用 self.skin_sigma

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
        # 1) 像素占比 (count) 与 视觉显著性 (salience) 采用双线性合并:
        #      combined = a*x + b*y + c*x*y
        #    其中 a=weight_count_norm, b=weight_chroma_norm, c=weight_bilinear
        #    c=0 时退化为纯加算 (旧行为);
        #    c>0 时: 面积大且显著性高的颜色获得额外加成,
        #            极小面积但高显著的颜色被抑制 (避免噪点上位)。
        # 2) 亮度均匀性 (variance) 与 空间中心距离 (center) 保持线性合并。
        a = self.weights[0]              # weight_count 归一化值
        b = self.weights[3]              # weight_chroma 归一化值
        c = self.weight_bilinear
        bilinear = (
            a * count_scores
            + b * salience_scores
            + c * count_scores * salience_scores
        )
        final_scores = (
            bilinear
            + self.weights[1] * variance_scores
            + self.weights[2] * center_scores
        )

        # --- 5. 肤色惩罚 (微调，乘性折扣) ---
        eff_penalty = self.skin_penalty_max if skin_penalty_max is None else float(skin_penalty_max)
        if eff_penalty > 0:
            eff_ref = self.skin_ref_lab if skin_ref_lab is None else np.asarray(skin_ref_lab, dtype=np.float64)
            eff_sigma = self.skin_sigma if skin_sigma is None else np.asarray(skin_sigma, dtype=np.float64)
            skin_sim = self.compute_skin_similarity(centers_lab, eff_ref, eff_sigma)
            final_scores = final_scores * (1.0 - skin_sim * eff_penalty)
            final_scores = np.clip(final_scores, 0, None)

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
                   img_h: int, img_w: int,
                   skin_penalty_max: Optional[float] = None,
                   skin_ref_lab: Optional[np.ndarray] = None,
                   skin_sigma: Optional[np.ndarray] = None) -> Tuple[RegionResult, Optional[dict]]:
    """
    分析单个区域（前景或背景）的主色。肤色惩罚始终使用 fixed 模式
    (固定的 Lab 参考中心 + 固定 sigma)，不存在动态提取路径。

    Args:
        region_name: 区域名称
        lab_pixels: (N, 3) Lab像素值
        pixel_coords: (N, 2) 像素坐标 [y, x]
        clusterer: 聚类器
        scorer: 评分器
        img_h: 图像高度
        img_w: 图像宽度
        skin_penalty_max: 覆盖 scorer 默认的肤色惩罚上限 (0~1)。0 关闭。
        skin_ref_lab: 覆盖 scorer 默认的肤色参考 (Lab, 3,)。
        skin_sigma: 覆盖 scorer 默认的肤色标准差 (Lab, 3,)。

    Returns:
        (RegionResult, skin_info)：
        - RegionResult: 区域主色结果
        - skin_info: 本次实际使用的肤色信息 dict
            { 'mode': 'off'/'fixed',
              'ref_lab': (3,), 'sigma': (3,),
              'penalty_max': float }
    """
    if len(lab_pixels) == 0:
        # 空区域
        empty_dominant = MainColor(
            lab=np.zeros(3), rgb=np.zeros(3),
            hex_color="#000000", score=0.0,
            proportion=0.0, cluster_id=-1
        )
        region_result = RegionResult(
            region_name=region_name,
            main_colors=[],
            dominant_color=empty_dominant
        )
        return region_result, {
            'mode': 'off', 'penalty_max': 0.0,
            'ref_lab': None, 'sigma': None,
        }

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

    # 解析肤色参数：调用方显式 > scorer 默认；不区分 fixed/dynamic
    eff_skin_penalty_max = (
        scorer.skin_penalty_max if skin_penalty_max is None
        else float(max(0.0, skin_penalty_max))
    )
    eff_skin_ref = skin_ref_lab
    eff_skin_sigma = skin_sigma
    skin_mode_used = 'fixed' if eff_skin_penalty_max > 0 else 'off'

    # 关键：记录本次实际使用的肤色参考 (含 scorer 默认值回退)，
    # 这样下游的 main.py 才能在控制台正确回显每个候选色的折扣。
    if eff_skin_penalty_max > 0:
        if eff_skin_ref is None:
            eff_skin_ref = scorer.skin_ref_lab
        if eff_skin_sigma is None:
            eff_skin_sigma = scorer.skin_sigma

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
        skin_penalty_max=eff_skin_penalty_max,
        skin_ref_lab=eff_skin_ref,
        skin_sigma=eff_skin_sigma,
    )

    dominant = main_colors[0] if main_colors else MainColor(
        lab=np.zeros(3), rgb=np.zeros(3),
        hex_color="#000000", score=0.0,
        proportion=0.0, cluster_id=-1
    )

    region_result = RegionResult(
        region_name=region_name,
        main_colors=main_colors[:5],  # 最多输出5个主色
        dominant_color=dominant
    )

    skin_info = {
        'mode': skin_mode_used,
        'penalty_max': eff_skin_penalty_max,
        'ref_lab': (
            np.asarray(eff_skin_ref, dtype=np.float64).copy()
            if eff_skin_ref is not None else None
        ),
        'sigma': (
            np.asarray(eff_skin_sigma, dtype=np.float64).copy()
            if eff_skin_sigma is not None else None
        ),
    }
    return region_result, skin_info
