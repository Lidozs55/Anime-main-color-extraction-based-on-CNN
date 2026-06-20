"""
Lab空间加权像素聚类模块。

核心思路：
  - 使用 Mini-Batch K-Means 提高效率
  - 通过缩放 Lab 维度实现"颜色优先、亮度辅助"的聚类策略
  - 可选择完全忽略亮度（纯色度聚类）或低权重保留亮度
"""
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from dataclasses import dataclass


@dataclass
class ClusterResult:
    """聚类结果"""
    labels: np.ndarray               # (N,) 每个像素的聚类标签
    centers_lab: np.ndarray          # (k, 3) 聚类中心 (标准Lab)
    n_pixels: np.ndarray             # (k,) 每个聚类的像素数
    proportions: np.ndarray          # (k,) 像素占比


class LabClusterer:
    """
    Lab空间聚类器。
    """

    def __init__(self, n_clusters: int = 8,
                 color_weight: float = 1.0,
                 lightness_weight: float = 0.2,
                 a_boost: float = 1.1,
                 batch_size: int = 1024,
                 random_state: int = 42):
        """
        Args:
            n_clusters: 聚类数
            color_weight: 颜色通道(a/b)的基权重倍数
            lightness_weight: 亮度通道(L)的权重倍数
                调低此值可降低亮度差异在聚类中的影响，使相近色相不同明度的颜色
                更容易被归为一类。推荐值 0.2。
            a_boost: a* 通道的额外放大倍数（红绿轴强化）
                人类对红绿色差更敏感，适当放大 a* 通道可提升红绿色差的分辨率。
                推荐值 1.1。
            batch_size: Mini-Batch大小
            random_state: 随机种子
        """
        self.n_clusters = n_clusters
        self.color_weight = color_weight
        self.lightness_weight = lightness_weight
        self.a_boost = a_boost
        self.batch_size = batch_size
        self.random_state = random_state

    def _scale_lab(self, lab_pixels: np.ndarray) -> np.ndarray:
        """
        对 Lab 像素进行加权缩放。
        缩放后的距离 d² = w_L²*ΔL² + (w_ab*a_boost)²*Δa² + w_ab²*Δb²

        通过调低 w_L、适当放大 a* 通道，实现"颜色优先、红绿更敏感"的聚类策略。

        Returns: (N, 3) 缩放后的特征
        """
        scaled = lab_pixels.copy()
        scaled[:, 0] *= self.lightness_weight       # L 通道：降低权重，≈0.2
        scaled[:, 1] *= self.color_weight * self.a_boost  # a* 通道：强化红绿
        scaled[:, 2] *= self.color_weight            # b* 通道：标准权重
        return scaled

    def _unscale_centers(self, scaled_centers: np.ndarray) -> np.ndarray:
        """将缩放后的聚类中心还原为标准Lab值"""
        centers = scaled_centers.copy()
        if self.lightness_weight > 0:
            centers[:, 0] /= self.lightness_weight
        else:
            centers[:, 0] = 0
        centers[:, 1] /= (self.color_weight * self.a_boost)
        centers[:, 2] /= self.color_weight
        return centers

    def fit(self, lab_pixels: np.ndarray) -> ClusterResult:
        """
        对Lab像素执行聚类。

        Args:
            lab_pixels: (N, 3) 标准Lab值, L[0,100], a/b[-128,127]

        Returns:
            ClusterResult
        """
        if len(lab_pixels) == 0:
            return ClusterResult(
                labels=np.array([], dtype=int),
                centers_lab=np.zeros((0, 3)),
                n_pixels=np.array([], dtype=int),
                proportions=np.array([], dtype=float)
            )

        # 过滤掉可能的无效像素
        valid = ~(np.isnan(lab_pixels).any(axis=1) | np.isinf(lab_pixels).any(axis=1))
        lab_pixels = lab_pixels[valid]

        n_samples = len(lab_pixels)
        effective_k = min(self.n_clusters, n_samples)

        if effective_k < 2:
            # 只有一个聚类或没有数据
            center = lab_pixels.mean(axis=0) if n_samples > 0 else np.zeros(3)
            return ClusterResult(
                labels=np.zeros(n_samples, dtype=int),
                centers_lab=center.reshape(1, 3),
                n_pixels=np.array([n_samples]),
                proportions=np.array([1.0])
            )

        # 缩放Lab特征
        scaled_pixels = self._scale_lab(lab_pixels)

        # Mini-Batch K-Means
        kmeans = MiniBatchKMeans(
            n_clusters=effective_k,
            batch_size=min(self.batch_size, n_samples),
            random_state=self.random_state,
            n_init=3,
            max_iter=100,
            max_no_improvement=10,
            reassignment_ratio=0.01,
            verbose=0
        )
        labels = kmeans.fit_predict(scaled_pixels)

        # 统计每个聚类
        unique_labels, counts = np.unique(labels, return_counts=True)
        proportions = counts / n_samples

        # 恢复聚类中心到标准Lab
        centers_lab = self._unscale_centers(kmeans.cluster_centers_)

        # 当 lightness_weight=0 时, L* 被缩放到0导致信息丢失
        # 用每个聚类中像素的原始 L* 均值重新计算（向量化）
        if self.lightness_weight == 0:
            l_sums = np.bincount(labels, weights=lab_pixels[:, 0], minlength=effective_k)
            # 先索引再赋值，避免形状不匹配
            for_ci = np.arange(effective_k)  # [0, 1, ..., effective_k-1]
            centers_lab[for_ci, 0] = l_sums / np.maximum(
                np.bincount(labels, minlength=effective_k).astype(np.float64), 1)

        # 仅保留有像素分配的聚类中心（按 unique_labels 顺序）
        k_actual = len(unique_labels)
        centers_lab = centers_lab[unique_labels]

        # 按像素数从大到小排序
        sort_idx = np.argsort(-counts)
        centers_lab = centers_lab[sort_idx]
        counts = counts[sort_idx]
        proportions = proportions[sort_idx]

        # 重新映射标签（向量化：避免 Python 字典查找）
        old_to_new = np.full(effective_k, -1, dtype=np.int32)
        old_to_new[unique_labels[sort_idx]] = np.arange(k_actual)
        labels = old_to_new[labels]

        return ClusterResult(
            labels=labels,
            centers_lab=centers_lab,
            n_pixels=counts,
            proportions=proportions
        )
