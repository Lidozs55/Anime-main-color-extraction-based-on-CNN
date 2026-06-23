"""
阴影去除模块 — 经典 Lab 空间方法(色差约束连通块 + 目标L + 线性混合 + ab 补偿,无羽化,无高光)。

集成位置
  - 教师管线:  graphcolor/pipeline.GraphColorPipeline.process()
               load_and_resize 之后、segment 之前调用
  - 学生推理:  student/preview.py, cv2.imread 之后立即调用
  - 学生训练:  student/dataset.ColorDataset.__getitem__ 同步应用
               (避免 train/inference 的 domain shift)

处理流水线
  BGR→Lab → 检测 mask → 形态学清理 → 色差约束连通块 → 逐块目标L → 线性混合 → 硬掩码 → ab 补偿

设计要点
  - 无羽化: 羽化在阴影/非阴影交界处产生"灰边"伪影;边缘过渡由下游"目标L"思路处理(职责分离)。
  - 色差约束连通块: mask 内像素按 RGB 色差约束(相邻像素 BGR max diff < color_threshold)
    划分为连通块,每块独立计算 median L,再套用 target_L 公式。
    保证浅阴影和深阴影即使空间相邻,只要色差够大就分属不同块,不会互相干扰。
  - 零模型文件、零下载、零额外依赖。
  - 不改变 a*/b* 方向(hue),只改 magnitude。
  - 软阴影/自阴影检测能力有限;极暗场景(L_illum < 25)不强行补偿;不处理高光。
"""
from typing import Optional
import cv2
import numpy as np


# ──────────────────────────────────────────────────────────────────────
# 色差约束连通块: 在 mask 内按 RGB 色差约束做连通域标记 (Union-Find)
# ──────────────────────────────────────────────────────────────────────
def _color_constrained_labeling(shadow_mask: np.ndarray,
                                 bgr: np.ndarray,
                                 color_threshold: float) -> tuple:
    """在 shadow_mask 内按 RGB 色差约束做连通域标记(光栅扫描 + 等效标签合并)。

    策略: 逐行光栅扫描,水平连通(左邻色差 < 阈值)继承同标签;垂直连通(上邻色差 < 阈值)
    记录标签等价关系;最后用打平后的等效标签表做 vectorized 赋值。
    避免 per-edge Union-Find 的 Python while 循环瓶颈。

    两个相邻阴影像素属于同一连通块,当且仅当它们的 BGR max 通道差
    小于 color_threshold。
    """
    h, w = shadow_mask.shape

    if not shadow_mask.any():
        return np.zeros((h, w), dtype=np.int32), 0

    bgr_i = bgr.astype(np.int16)
    ct = float(color_threshold)

    # Precompute 色差(向量化)
    h_diff = np.max(np.abs(bgr_i[:, 1:] - bgr_i[:, :-1]), axis=2)   # (h, w-1)
    v_diff = np.max(np.abs(bgr_i[1:, :] - bgr_i[:-1, :]), axis=2)  # (h-1, w)

    # Union-Find (仅对标签 ID,非像素)
    parent: dict[int, int] = {}

    def _find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def _union(x: int, y: int) -> None:
        rx, ry = _find(x), _find(y)
        if rx != ry:
            parent[rx] = ry

    # Pass 1: 光栅扫描分配临时标签
    labels = np.zeros((h, w), dtype=np.int32)
    next_label = 0
    _prev_row_labels = np.zeros(w, dtype=np.int32)  # 缓存上一行标签,避免 labels[y-1] 索引开销

    for y in range(h):
        row_mask = shadow_mask[y]
        _this_row = labels[y]

        for x in range(w):
            if not row_mask[x]:
                continue

            left_ok = (x > 0 and row_mask[x-1] and h_diff[y, x-1] < ct)
            up_ok = (y > 0 and shadow_mask[y-1, x] and v_diff[y-1, x] < ct)

            if left_ok:
                label = _this_row[x-1]
                if up_ok:
                    other = _prev_row_labels[x]
                    if label != other:
                        _union(label, other)
                _this_row[x] = label
            elif up_ok:
                _this_row[x] = _prev_row_labels[x]
            else:
                next_label += 1
                _this_row[x] = next_label
                parent[next_label] = next_label

        _prev_row_labels = _this_row

    if next_label == 0:
        return labels, 0

    # Pass 2: 打平所有标签的根
    max_label = next_label
    all_roots = np.arange(max_label + 1, dtype=np.int32)
    for lab in range(1, max_label + 1):
        all_roots[lab] = _find(lab)

    # Pass 3: 连续编号 1..n_labels
    unique_roots = np.unique(all_roots[1:])
    root_to_new = np.zeros(max_label + 1, dtype=np.int32)
    for i, r in enumerate(unique_roots):
        root_to_new[r] = i + 1
    n_labels = len(unique_roots)

    # Pass 4: Vectorized 赋值
    shadow_idx = np.flatnonzero(shadow_mask)
    final = root_to_new[all_roots[labels.ravel()[shadow_idx]]]
    out = np.zeros((h, w), dtype=np.int32)
    out.ravel()[shadow_idx] = final
    return out, n_labels


# ──────────────────────────────────────────────────────────────────────
# 阴影去除:目标 L + 线性混合 + ab 补偿,无羽化
# ──────────────────────────────────────────────────────────────────────

def _filter_components_by_ab_consistency(labels, n_labels, lab, L, shadow_mask,
                                          ab_consistency_threshold=20.0):
    """逐连通块 ab 比例一致性检查,排除异色物体误判为阴影。

    对每个连通块:
      1. 膨胀 mask(5x5 椭圆,3 次)取外环
      2. 组件均值 Lab 与环均值 Lab
      3. 预测 ab = ring_ab × (comp_L / ring_L), 偏差 > threshold → 排除

    优化策略:
      - 用 np.bincount 一次性计算所有组件的均值(免去逐个 mask 的 O(N) 扫描)
      - 对满足大小门槛的组件,用边界框约束缩小 dilation 区域(避免在 512x512 上逐个 dilate)

    核心假设: 真实阴影的 ab 随 L 同比例缩小,异色物体则不然。
    """
    if n_labels == 0 or ab_consistency_threshold <= 0:
        return shadow_mask, False

    H, W = labels.shape
    a_signed = lab[..., 1] - 128.0
    b_signed = lab[..., 2] - 128.0

    # ── 向量化: 一次性计算所有组件的像素数和均值 ──
    flat = labels.ravel()
    flat_L = L.ravel()
    flat_a = a_signed.ravel()
    flat_b = b_signed.ravel()

    cnt = np.bincount(flat, minlength=n_labels + 1)          # (n_labels+1,)
    sum_L = np.bincount(flat, weights=flat_L, minlength=n_labels + 1)
    sum_a = np.bincount(flat, weights=flat_a, minlength=n_labels + 1)
    sum_b = np.bincount(flat, weights=flat_b, minlength=n_labels + 1)
    # cnt[0] = 非阴影(忽略)
    mean_L = np.zeros(n_labels + 1, dtype=np.float64)
    mean_a = np.zeros(n_labels + 1, dtype=np.float64)
    mean_b = np.zeros(n_labels + 1, dtype=np.float64)
    ok = cnt > 0
    mean_L[ok] = sum_L[ok] / cnt[ok]
    mean_a[ok] = sum_a[ok] / cnt[ok]
    mean_b[ok] = sum_b[ok] / cnt[ok]

    # ── 筛选出需检查的标签(>=9 像素) ──
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    rejected = []
    # 跳过 label 0(非阴影)
    check_labels = np.flatnonzero(cnt[1:] >= 9) + 1

    for label_id in check_labels:
        # 边界框约束: 只取组件所在区域 + 6px padding(3 次 dilation 最大膨胀量)
        ys, xs = np.where(labels == label_id)
        y1 = max(0, int(ys.min()) - 8)
        y2 = min(H, int(ys.max()) + 9)
        x1 = max(0, int(xs.min()) - 8)
        x2 = min(W, int(xs.max()) + 9)

        sub = labels[y1:y2, x1:x2]
        sub_mask = (sub == label_id).astype(np.uint8)

        expanded = cv2.dilate(sub_mask, kernel, iterations=3).astype(bool)
        ring = expanded & (sub != label_id)
        n_ring = ring.sum()
        if n_ring < 16:
            continue

        # 环的均值
        ring_L = L[y1:y2, x1:x2][ring].mean()
        ring_a = a_signed[y1:y2, x1:x2][ring].mean()
        ring_b = b_signed[y1:y2, x1:x2][ring].mean()

        comp_L = mean_L[label_id]
        comp_a = mean_a[label_id]
        comp_b = mean_b[label_id]

        L_ratio = comp_L / max(ring_L, 1e-6)
        ab_error = (abs(comp_a - ring_a * L_ratio)
                    + abs(comp_b - ring_b * L_ratio))

        if ab_error > ab_consistency_threshold:
            rejected.append(label_id)

    if not rejected:
        return shadow_mask, False

    new_mask = shadow_mask.copy()
    for lid in rejected:
        new_mask &= (labels != lid)
    return new_mask, True


def _remove_shadow_lab(bgr: np.ndarray,
                       sigma: int,
                       threshold: float,
                       dark_object_ratio: float = 0.20,
                       ab_compensation_alpha: float = 0.3,
                       use_morphology: bool = True,
                       morph_kernel_size: int = 3,
                       target_l_offset: float = 0.08,
                       target_l_gain: float = 1.2,
                       shadow_blend: float = 0.5,
                       color_threshold: float = 15.0,
                       ab_consistency_threshold: float = 25.0,
                       fast_mode: bool = False) -> np.ndarray:
    """Lab 空间阴影去除(目标 L + 线性混合 + ab 补偿,纯 numpy + opencv,无羽化)。

    算法步骤
      1) BGR → Lab
      2) 大 σ 高斯估计光照分量 L_illum(只用于检测)
      3) 多条件阴影检测(diff + 周围够亮 + 非暗物体)
      4) 形态学清理(OPEN 去噪 + 1 次 DILATE 轻膨胀填洞)
      5) [标准模式] 色差约束连通块: mask 内按 BGR max 通道差 < color_threshold 做连通域标记
      6) [标准模式] 逐组件 ab 一致性检查: 排除异色物体误判
      7) [标准/快速] 目标 L: 标准模式按块独立计算,快速模式全局统一计算
      8) [标准/快速] 线性混合 + 下限保护
      9) [标准/快速] 硬掩码 L 修正
      10) [标准/快速] ab 补偿

    Args:
        bgr: 输入 BGR uint8 图
        sigma: 高斯 σ(光照估计用)
        threshold: L_illum − L > 该值才视为阴影(默认 3.0)
        dark_object_ratio: 暗物体过滤比例
        ab_compensation_alpha: ab 补偿强度(0=关闭)
        use_morphology: 是否对 mask 做开运算+轻膨胀
        morph_kernel_size: 形态学核大小
        target_l_offset: 目标 L 的加性偏移(0-1 空间)
        target_l_gain: 目标 L 的乘性增益
        shadow_blend: 阴影区线性混合比例
        color_threshold: 色差约束连通块阈值(标准模式)
        ab_consistency_threshold: 逐组件 ab 一致性阈值(标准模式)
        fast_mode: 若为 True,跳过连通块标记和 ab 一致性检查,全局计算目标 L,
                  速度更快但精度略降(适用于对速度敏感的推理场景,如 preview.py)

    Returns:
        与输入同形状、同 dtype 的 BGR uint8 图像。
    """
    H, W = bgr.shape[:2]
    if H < 16 or W < 16:
        return bgr

    # 1) BGR → Lab
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab).astype(np.float32)
    L = lab[..., 0]

    # 2) 大 σ 高斯估计光照分量 (降采样加速: 2x 缩小 → 小核模糊 → 2x 放大)
    #    L_illum 只用于阴影检测,降采样引入的微小误差被阈值 margin 吸收。
    h_small, w_small = max(16, H // 2), max(16, W // 2)
    L_small = cv2.resize(L, (w_small, h_small))
    L_illum_small = cv2.GaussianBlur(L_small, (0, 0),
                                     sigmaX=sigma / 2.0, sigmaY=sigma / 2.0)
    L_illum = cv2.resize(L_illum_small, (W, H))

    # 3) 多条件阴影 mask:
    #    a) L 显著低于 L_illum(阈值 3.0,激进以捕获浅阴影)
    #    b) 周围不能太暗(L_illum > 15,允许偏暗图的浅阴影)
    #    c) L > ratio * L_illum(过滤黑物体,保留真实阴影;0.20 放宽)
    cond_diff = (L_illum - L) > threshold
    cond_illum = L_illum > 15.0
    cond_not_dark = L > dark_object_ratio * L_illum
    shadow_mask = cond_diff & cond_illum & cond_not_dark

    if not shadow_mask.any():
        return bgr

    # 4) 形态学清理: OPEN 去噪 + DILATE(1次)轻膨胀填洞
    #    (用 3×3 核保守,避免 CLOSE 把 mask 扩展 2-3 像素)
    if use_morphology:
        # Fast mode 强制更保守的核(3×3,不开 1 次 DILATE 之外的额外扩张),
        # 防止快速模式因跳过 ab 一致性检查而把暗物体误判为阴影
        if fast_mode:
            k = 3
        else:
            k = max(3, int(morph_kernel_size) | 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask_u8 = cv2.morphologyEx(
            (shadow_mask.astype(np.uint8) * 255),
            cv2.MORPH_OPEN, kernel,
        )
        if not fast_mode:
            # 标准模式: 1 次 DILATE 轻膨胀填洞
            mask_u8 = cv2.dilate(mask_u8, kernel, iterations=1)
        shadow_mask = mask_u8 > 127
        if not shadow_mask.any():
            return bgr

    if fast_mode:
        # ── 快速模式: 跳过连通块标记和 ab 一致性检查 ──
        # 修复参数 (target_l_offset / target_l_gain / shadow_blend / ab_compensation_alpha)
        # **与标准模式完全一致**,只调整**检测段**使其更保守:
        #   - threshold 2.0(比标准 3.0 更小)→ 真实阴影 L_illum−L > 2 即可
        #   - dark_object_ratio 0.35(比标准 0.20 更严)→ 排除更多暗物体
        #   - cond_illum 18.0(比标准 15.0 略严)→ 周围更亮才允许检测
        # 目的: 抵消跳过 ab 一致性检查带来的"把暗物体当阴影"风险
        det_threshold = min(threshold, 2.0)
        det_dark_ratio = max(dark_object_ratio, 0.35)
        det_min_illum = 18.0

        cond_diff = (L_illum - L) > det_threshold
        cond_illum = L_illum > det_min_illum
        cond_not_dark = L > det_dark_ratio * L_illum
        det_mask = cond_diff & cond_illum & cond_not_dark
        if not det_mask.any():
            return bgr
        # 用更保守的检测结果替换原 shadow_mask
        shadow_mask = det_mask

        # 直接取阴影 mask 的全局均值作为目标 L,统一应用于整个 shadow mask
        target_L_arr = np.zeros_like(L)
        shadow_mean_L = L[shadow_mask].mean()
        target_val = target_l_offset * 255.0 + target_l_gain * shadow_mean_L
        target_L_arr[shadow_mask] = target_val
        target_L_arr = np.clip(target_L_arr, 0.0, 255.0)
    else:
        # 5) 色差约束连通块: 相邻像素 BGR max 通道差 < color_threshold 才归入同一连通块
        #    确保浅阴影和深阴影即使空间相邻也不会被合并到同一块
        labels, n_labels = _color_constrained_labeling(shadow_mask, bgr, color_threshold)

        # 6) 逐组件 ab 一致性检查: 组件级 O(N) 检查,不引入额外高斯模糊,
        #    跨颜色边界阴影不受影响(每种颜色半区形成独立组件)
        if n_labels > 0:
            shadow_mask, _ = _filter_components_by_ab_consistency(
                labels, n_labels, lab, L, shadow_mask, ab_consistency_threshold,
            )
            if not shadow_mask.any():
                return bgr
            labels, n_labels = _color_constrained_labeling(shadow_mask, bgr, color_threshold)

        # 7) 逐块目标 L: 每块独立计算 mean L,套用 target_L 公式
        #    用 np.bincount 向量化逐块均值替代 sort+groupby+for 循环
        #    (色差约束连通块内像素颜色相近,均值 ≈ 中位数)
        target_L_arr = np.zeros_like(L)
        if n_labels > 0:
            shadow_flat_labels = labels[shadow_mask]
            shadow_flat_L = L[shadow_mask]
            sum_per_label = np.bincount(shadow_flat_labels, weights=shadow_flat_L,
                                        minlength=n_labels + 2)[1:]
            cnt_per_label = np.bincount(shadow_flat_labels,
                                        minlength=n_labels + 2)[1:]
            mean_per_label = sum_per_label / np.maximum(cnt_per_label, 1)
            flat_target = (target_l_offset * 255.0
                           + target_l_gain * mean_per_label[shadow_flat_labels - 1])
            shadow_indices = np.flatnonzero(shadow_mask)
            target_L_arr.ravel()[shadow_indices] = flat_target
        target_L_arr = np.clip(target_L_arr, 0.0, 255.0)

    # 8) 线性混合 + 下限保护(修复"亮边变暗"伪影):
    #    若原 L > target_L,L_blend < L,max 保护让该像素保持原 L
    L_blend = (1.0 - shadow_blend) * L + shadow_blend * target_L_arr
    L_new = np.maximum(L, L_blend)

    # 9) 硬掩码 L 修正(阴影区填 L_new,区外保持原 L;无羽化)
    L_final = np.where(shadow_mask, L_new, L)

    # 10) ab 补偿: L 改后感知饱和度变化
    #     per_pixel_boost = L_new[px] / L[px],掩码内 C_new = C_old × boost^α
    #     必须先减 128 到 signed Lab 再缩放(直接对 uint8 缩放会暴增 chroma)
    if ab_compensation_alpha > 0:
        per_pixel_boost = L_new / np.maximum(L, 1e-3)
        ab_scale = np.where(
            shadow_mask,
            np.power(np.maximum(per_pixel_boost, 1e-6), ab_compensation_alpha),
            1.0,
        )
        a_final = np.clip((lab[..., 1] - 128.0) * ab_scale + 128.0, 0.0, 255.0)
        b_final = np.clip((lab[..., 2] - 128.0) * ab_scale + 128.0, 0.0, 255.0)
    else:
        a_final = lab[..., 1]
        b_final = lab[..., 2]

    lab_out = lab.copy()
    lab_out[..., 0] = L_final
    lab_out[..., 1] = a_final
    lab_out[..., 2] = b_final
    return cv2.cvtColor(lab_out.astype(np.uint8), cv2.COLOR_Lab2BGR)


# ──────────────────────────────────────────────────────────────────────
# ShadowRemover:阴影去除统一入口(纯阴影,无高光)
# ──────────────────────────────────────────────────────────────────────
class ShadowRemover:
    """阴影去除器(纯经典 Lab 空间方法,无神经网络,无羽化,无高光)。

    默认参数为动漫/插画场景下的推荐值;真实照片/3D 渲染可适当调小
    ``shadow_threshold``、调大 ``sigma_ratio``、关闭 ``use_morphology``。

    .. note::
        所有可调参数的**规范配置源**位于
        :py:data:`graphcolor.pipeline.DEFAULT_CONFIG["shadow_removal_params"]`。
        此处的构造函数签名默认值仅作为独立使用时的回退值,
        若通过 ``GraphColorPipeline`` 调用,参数以 ``pipeline`` 的配置为准。

    .. note::
        ``fast_mode`` 仅影响独立调用时的行为；若通过 ``GraphColorPipeline``
        调用,忽略此参数(教师管线始终使用标准模式)。
    """

    def __init__(
        self,
        # 总开关
        enabled: bool = True,

        # ── 阴影检测与修正 ──
        sigma_ratio: float = 0.10,                 # 高斯 σ 与图片短边的比值
        shadow_threshold: float = 3.0,             # L_illum − L > 该值才视为阴影
        dark_object_ratio: float = 0.20,           # L > ratio × L_illum 才视为潜在阴影

        use_morphology: bool = True,
        morph_kernel_size: int = 5,

        target_l_offset: float = 0.08,             # 目标 L 加性偏移
        target_l_gain: float = 1.2,                # 目标 L 乘性增益
        shadow_blend: float = 0.5,                 # 线性混合比例
        color_threshold: float = 15.0,             # 色差约束阈值

        ab_compensation_alpha: float = 0.3,        # ab 补偿强度
        ab_consistency_threshold: float = 20.0,    # 逐组件 ab 一致性阈值

        # 快速模式
        fast_mode: bool = False,                   # 跳过连通块+ab检查,全局目标L(更快但精度略降)
    ):
        for key, value in locals().items():
            if key != 'self':
                setattr(self, key, value)

    def _compute_sigma(self, bgr: np.ndarray) -> int:
        """根据短边和 sigma_ratio 计算高斯 σ。"""
        H, W = bgr.shape[:2]
        return max(15, int(min(H, W) * self.sigma_ratio))

    def remove(self, bgr: np.ndarray) -> np.ndarray:
        """对单张 BGR uint8 图像做阴影去除,返回同形状的 BGR uint8。

        输入 None / 空 / 极小图 (<16x16) 时直接返回原图。
        """
        if not self.enabled or bgr is None or bgr.size == 0:
            return bgr
        H, W = bgr.shape[:2]
        if H < 16 or W < 16:
            return bgr
        return _remove_shadow_lab(
            bgr,
            sigma=self._compute_sigma(bgr),
            threshold=self.shadow_threshold,
            dark_object_ratio=self.dark_object_ratio,
            ab_compensation_alpha=self.ab_compensation_alpha,
            use_morphology=self.use_morphology,
            morph_kernel_size=self.morph_kernel_size,
            target_l_offset=self.target_l_offset,
            target_l_gain=self.target_l_gain,
            shadow_blend=self.shadow_blend,
            color_threshold=self.color_threshold,
            ab_consistency_threshold=self.ab_consistency_threshold,
            fast_mode=self.fast_mode,
        )


# ──────────────────────────────────────────────────────────────────────
# 批量预览: 对 img/ 下所有图片应用阴影去除,输出至 outputs/shadow_preview/
# ──────────────────────────────────────────────────────────────────────
def shadow_preview(img_dir: str = None,
                   output_dir: str = None,
                   remover: ShadowRemover = None,
                   max_size: int = None) -> None:
    """批量应用阴影去除并输出预览图。

    Args:
        img_dir: 输入图片目录（默认: 项目根目录下的 img/）
        output_dir: 输出目录（默认: 项目根目录下的 outputs/shadow_preview/）
        remover: 阴影去除器实例（默认: 使用 ShadowRemover()）
        max_size: 处理前 resize 的最长边（默认: None=原尺寸;建议 512 加速大图）
    """
    import os, glob, time
    from pathlib import Path

    script_dir = Path(__file__).resolve().parent.parent  # graphcolor/..
    img_dir = img_dir or os.path.join(script_dir, 'img')
    output_dir = output_dir or os.path.join(script_dir, 'outputs', 'shadow_preview')

    exts = ('*.jpg', '*.jpeg', '*.png', '*.bmp')
    paths = []
    for e in exts:
        paths.extend(glob.glob(os.path.join(img_dir, e)))
        paths.extend(glob.glob(os.path.join(img_dir, e.upper())))
    paths = sorted(set(p.lower() for p in paths))

    if not paths:
        print(f"在 {img_dir} 中未找到图片")
        return

    os.makedirs(output_dir, exist_ok=True)

    sr = remover or ShadowRemover()
    total_t, ok, skip = 0.0, 0, 0

    print(f"阴影去除预览 — 处理 {len(paths)} 张图片")
    print(f"  输入: {img_dir}")
    print(f"  输出: {output_dir}")
    if max_size:
        print(f"  resize: 最长边 ≤ {max_size}px (加速大图处理)")
    print()

    for i, p in enumerate(paths):
        name = os.path.basename(p)
        img = cv2.imread(p)
        if img is None:
            print(f"  [{i+1}/{len(paths)}] 跳过: {name}")
            skip += 1
            continue

        if max_size and max(img.shape[:2]) > max_size:
            scale = max_size / max(img.shape[:2])
            new_size = (int(img.shape[1] * scale), int(img.shape[0] * scale))
            img = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)

        t0 = time.perf_counter()
        out = sr.remove(img)
        ms = (time.perf_counter() - t0) * 1000
        total_t += ms

        out_path = os.path.join(output_dir, name)
        cv2.imwrite(out_path, out)
        print(f"  [{i+1}/{len(paths)}] {ms:6.1f}ms  {name}")
        ok += 1

    print(f"\n完成: {ok} 张成功, {skip} 张跳过")
    print(f"总耗时: {total_t/1000:.1f}s  平均: {total_t/ok:.0f}ms/张")


# ──────────────────────────────────────────────────────────────────────
# 单元测试
# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if len(sys.argv) >= 2 and sys.argv[1] == "bench":
        # 简单性能 benchmark
        rng = np.random.default_rng(0)
        img = rng.integers(120, 200, (512, 512, 3), dtype=np.uint8)
        img[128:384, 128:384] //= 2
        r = ShadowRemover()
        import time
        t0 = time.perf_counter()
        for _ in range(20):
            _ = r.remove(img)
        ms = (time.perf_counter() - t0) / 20 * 1000
        print(f"512x512 单张耗时(无高光): {ms:.2f}ms")
    elif len(sys.argv) >= 2 and sys.argv[1] == "preview":
        # 批量预览
        max_size = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 512
        shadow_preview(max_size=max_size)
    else:
        # 正确性测试
        r = ShadowRemover()

        # 辅助函数: 手动模拟算法, 计算期望 L_new
        def _expected_l_new(img_bgr, remover, roi=None):
            """手动模拟完整流水线, 返回期望 L_new (roi 区域均值) 或全图."""
            lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab).astype(np.float32)
            L = lab[..., 0]
            sigma = remover._compute_sigma(img_bgr)
            L_illum = cv2.GaussianBlur(L, (0, 0), sigmaX=sigma, sigmaY=sigma)
            cond_diff = (L_illum - L) > remover.shadow_threshold
            cond_illum = L_illum > 25.0
            cond_not_dark = L > remover.dark_object_ratio * L_illum
            sm = cond_diff & cond_illum & cond_not_dark
            if sm.any() and remover.use_morphology:
                k = max(3, int(remover.morph_kernel_size) | 1)
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
                mu8 = cv2.morphologyEx((sm.astype(np.uint8) * 255), cv2.MORPH_OPEN, kernel)
                mu8 = cv2.dilate(mu8, kernel, iterations=1)
                sm = mu8 > 127
            if not sm.any():
                return L
            labels, n_labels = _color_constrained_labeling(sm, img_bgr, remover.color_threshold)
            target_L_arr = np.zeros_like(L)
            if n_labels > 0:
                flat_labels = labels[sm]
                flat_L = L[sm]
                sum_per_label = np.bincount(flat_labels, weights=flat_L,
                                            minlength=n_labels + 2)[1:]
                cnt_per_label = np.bincount(flat_labels,
                                            minlength=n_labels + 2)[1:]
                mean_per_label = sum_per_label / np.maximum(cnt_per_label, 1)
                flat_target = (remover.target_l_offset * 255.0
                               + remover.target_l_gain * mean_per_label[flat_labels - 1])
                shadow_idx = np.flatnonzero(sm)
                target_L_arr.ravel()[shadow_idx] = flat_target
            target_L_arr = np.clip(target_L_arr, 0.0, 255.0)
            L_blend = (1.0 - remover.shadow_blend) * L + remover.shadow_blend * target_L_arr
            L_new = np.maximum(L, L_blend)
            if roi is not None:
                return float(L_new[roi].mean())
            return L_new

        # T1: 阴影被提亮(色差约束连通块, 逐块 target_L)
        img = np.full((256, 256, 3), 200, dtype=np.uint8)
        img[64:192, 64:192] = (80, 80, 80)
        out = r.remove(img)
        l_before = cv2.cvtColor(img[64:192, 64:192], cv2.COLOR_BGR2Lab)[..., 0].mean()
        l_after = cv2.cvtColor(out[64:192, 64:192], cv2.COLOR_BGR2Lab)[..., 0].mean()
        l_exp = _expected_l_new(img, r, roi=(slice(64, 192), slice(64, 192)))
        print(f"[T1] 阴影区 L: {l_before:.1f} → {l_after:.1f}  (期望 L_new={l_exp:.1f})")
        assert l_after > l_before, "阴影应被提亮"
        assert abs(l_after - l_exp) < 3.0, \
            f"预期 L≈{l_exp:.1f}, 实际 {l_after:.1f}"

        # T2: 非阴影区应保持不变
        l_outside_before = cv2.cvtColor(img[:32, :32], cv2.COLOR_BGR2Lab)[..., 0].mean()
        l_outside_after = cv2.cvtColor(out[:32, :32], cv2.COLOR_BGR2Lab)[..., 0].mean()
        print(f"[T2] 非阴影区 L: {l_outside_before:.1f} → {l_outside_after:.1f}")
        assert abs(l_outside_after - l_outside_before) < 1.0

        # T3: 黑头发场景 → 暗物体被过滤
        img2 = np.full((256, 256, 3), 200, dtype=np.uint8)
        img2[64:192, 64:192] = (20, 20, 20)
        out2 = r.remove(img2)
        l_hair_before = cv2.cvtColor(img2[64:192, 64:192], cv2.COLOR_BGR2Lab)[..., 0].mean()
        l_hair_after = cv2.cvtColor(out2[64:192, 64:192], cv2.COLOR_BGR2Lab)[..., 0].mean()
        print(f"[T3] 黑矩形 L: {l_hair_before:.1f} → {l_hair_after:.1f}")
        assert l_hair_after < l_hair_before + 20, "黑物体应被过滤"

        # T4: 不同 target_l_gain 的行为(gain=1.5 应比 1.2 提亮更多)
        r_g15 = ShadowRemover(target_l_gain=1.5)
        r_g12 = ShadowRemover(target_l_gain=1.2)
        img_t4 = np.full((256, 256, 3), 200, dtype=np.uint8)
        img_t4[64:192, 64:192] = (80, 80, 80)
        l_g15 = cv2.cvtColor(r_g15.remove(img_t4)[64:192, 64:192],
                             cv2.COLOR_BGR2Lab)[..., 0].mean()
        l_g12 = cv2.cvtColor(r_g12.remove(img_t4)[64:192, 64:192],
                             cv2.COLOR_BGR2Lab)[..., 0].mean()
        print(f"[T4] gain=1.2 L={l_g12:.1f}, gain=1.5 L={l_g15:.1f}")
        assert l_g15 > l_g12 + 5, "gain=1.5 应比 1.2 提亮更多"
        ratio = l_g15 / l_g12
        assert 1.05 < ratio < 1.20, f"blend=0.5 提亮比例应≈1.10, 实际 {ratio:.3f}"

        # T5: ab 补偿 - 阴影区 a/b 缩放, hue 保持
        r_nocomp = ShadowRemover(ab_compensation_alpha=0.0)
        r_comp = ShadowRemover(ab_compensation_alpha=0.5)
        img_t5 = np.full((256, 256, 3), 200, dtype=np.uint8)
        img_t5[96:160, 96:160] = (50, 65, 80)
        out_nocomp = r_nocomp.remove(img_t5)
        out_comp = r_comp.remove(img_t5)
        cy, cx = 128, 128
        def signed_ab(region_bgr):
            lab = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2Lab)
            return (float(lab[..., 0].mean()),
                    float(lab[..., 1].mean()) - 128.0,
                    float(lab[..., 2].mean()) - 128.0)
        l_nc, a_nc_s, b_nc_s = signed_ab(out_nocomp[cy-3:cy+3, cx-3:cx+3])
        l_c, a_c_s, b_c_s = signed_ab(out_comp[cy-3:cy+3, cx-3:cx+3])
        L_ORIG = float(np.median(cv2.cvtColor(img_t5[96:160, 96:160], cv2.COLOR_BGR2Lab)[..., 0]))
        print(f"[T5] L_ORIG={L_ORIG:.1f}, L(不补偿)={l_nc:.1f}, L(补偿)={l_c:.1f}")
        assert l_c > L_ORIG + 5, f"阴影应被提亮, L 仅 {l_c-L_ORIG:.1f}"
        assert l_nc > L_ORIG + 5, f"阴影应被提亮, L 仅 {l_nc-L_ORIG:.1f}"
        assert abs(l_c - l_nc) < 3.0, f"补偿不应改变 L, 差 {l_c-l_nc:.1f}"
        mag_nc = (a_nc_s**2 + b_nc_s**2) ** 0.5
        mag_c = (a_c_s**2 + b_c_s**2) ** 0.5
        assert mag_c > mag_nc * 1.03, \
            f"α=0.5 补偿后色饱应提升≥3%, 实际 {mag_nc:.2f} → {mag_c:.2f}"
        import math
        hue_nc = math.degrees(math.atan2(b_nc_s, a_nc_s))
        hue_c = math.degrees(math.atan2(b_c_s, a_c_s))
        hue_diff = min(abs(hue_c - hue_nc), 360 - abs(hue_c - hue_nc))
        assert hue_diff < 5.0, f"hue 应保持不变, 差 {hue_diff:.2f}°"
        per_pixel_boost = l_c / L_ORIG
        expected_ratio = per_pixel_boost ** 0.5
        actual_ratio = mag_c / mag_nc
        assert 0.90 * expected_ratio < actual_ratio < 1.10 * expected_ratio, \
            f"色饱比例应≈{expected_ratio:.3f}, 实际 {actual_ratio:.3f}"

        # T6: α=0 等同关闭补偿
        r_alpha0 = ShadowRemover(ab_compensation_alpha=0.0)
        out_alpha0 = r_alpha0.remove(img_t5)
        _, a_a0_s, b_a0_s = signed_ab(out_alpha0[cy-3:cy+3, cx-3:cx+3])
        assert abs(a_a0_s - a_nc_s) < 1.0, f"α=0 等同不补偿, a 差 {a_a0_s-a_nc_s:.2f}"
        assert abs(b_a0_s - b_nc_s) < 1.0, f"α=0 等同不补偿, b 差 {b_a0_s-b_nc_s:.2f}"

        # T7: 浅阴影和深阴影分属不同连通块, 各自独立提亮
        # 构造: 深阴影 100x100 BGR(50) + 浅阴影 4x4 BGR(100)
        # BGR max diff = 50 > color_threshold(15) → 分属不同连通块
        img_t7 = np.full((256, 256, 3), 255, dtype=np.uint8)
        img_t7[78:178, 78:178] = (50, 50, 50)
        img_t7[170:174, 170:174] = (100, 100, 100)
        out_t7 = r.remove(img_t7)
        L_bright_before = cv2.cvtColor(img_t7[170:174, 170:174], cv2.COLOR_BGR2Lab)[..., 0].mean()
        L_bright_after = cv2.cvtColor(out_t7[170:174, 170:174], cv2.COLOR_BGR2Lab)[..., 0].mean()
        print(f"[T7] 浅阴影 4x4 L: {L_bright_before:.1f} → {L_bright_after:.1f} "
              f"(色差约束连通块分离, 独立 target_L)")
        assert L_bright_after > L_bright_before + 5, \
            f"浅阴影应被提亮, 实际 {L_bright_before:.1f}→{L_bright_after:.1f}"
        L_shadow_before = cv2.cvtColor(img_t7[120:130, 120:130], cv2.COLOR_BGR2Lab)[..., 0].mean()
        L_shadow_after = cv2.cvtColor(out_t7[120:130, 120:130], cv2.COLOR_BGR2Lab)[..., 0].mean()
        print(f"[T7] 深阴影区 L: {L_shadow_before:.1f} → {L_shadow_after:.1f} (应被提亮)")
        assert L_shadow_after > L_shadow_before + 5, "深阴影应被提亮"
        # 验证浅阴影和深阴影分属不同连通块(通过 target_L 不同来间接验证)
        bright_vs_dark_ratio = L_bright_after / L_shadow_after
        print(f"[T7] 浅阴影/深阴影 L_new 比值: {bright_vs_dark_ratio:.3f} "
              f"(>1.5 表明确实在不同连通块)")
        assert bright_vs_dark_ratio > 1.5, \
            f"浅阴影 target_L 应远高于深阴影(分属不同连通块), 比值 {bright_vs_dark_ratio:.3f}"

        print("OK: 阴影提亮 / 非阴影不变 / 黑物体过滤 / "
              "目标L gain 可调 / ab 补偿生效 / α=0 关闭 / hue 保持 / 比例符合 CIE 公式 / "
              "浅深阴影色差约束分离(连通块独立 target_L)")
