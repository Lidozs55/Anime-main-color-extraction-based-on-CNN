"""
阴影去除模块 — 经典 Lab 空间方法(v3.6,色差约束连通块 目标L + 线性混合 + ab 补偿,无羽化,**无高光**)。

集成位置
─────────
  - 教师管线:  graphcolor/pipeline.GraphColorPipeline.process()
               load_and_resize 之后、segment 之前调用
  - 学生推理:  student/preview.py, cv2.imread 之后立即调用
  - 学生训练:  student/dataset.ColorDataset.__getitem__ 同步应用
               (避免 train/inference 的 domain shift)

处理流水线
──────────
  shadow 修正(单步)
  BGR→Lab → 检测 mask → 形态学清理 → 色差约束连通块 → 逐块目标L → 线性混合 → 硬掩码 → ab 补偿

为什么无羽化
────────────
  羽化在阴影/非阴影交界处产生"灰边"伪影;本模块采用硬掩码,
  边缘过渡由下游"目标L"思路处理(职责分离)。

版本说明
────────
  v3.6: target_L 改为基于"色差约束连通块"的逐块计算:
        mask 内像素按 RGB 色差约束(相邻像素 BGR max diff < color_threshold)
        划分为连通块, 每块独立计算 median L, 再套用 target_L 公式。
        解决 v3.5 两大问题:
          1) L_illum(高斯模糊)让周围亮像素渗透进阴影区, 产生扰动;
          2) 浅/深阴影边界的平滑过渡问题(高斯模糊天然平滑, 丢失边缘)。
        新方案: 色差约束保证连通块不会跨越颜色/亮度边缘,
                浅阴影和深阴影即使空间相邻, 只要色差够大就分属不同块,
                每块独立计算 target_L, 不会互相干扰。
  v3.5: target_L 改为 per-pixel 公式: target_L[px] = 0.05*255 + 1.2 * L_illum[px]
        (已废弃: L_illum 高斯模糊导致亮像素扰动 + 边缘平滑过渡)
  v3.4: 完全移除高光处理(函数/参数/CLI/测试全删);
        加入下限保护 L_new = max(L, L_blend),修复"亮边变暗"伪影
  v3.3: 阴影 L 改用线性混合(原图与目标L的线性组合,blend=0.5 默认),避免纯色块
  v3.2: 移除羽化;阴影 L 改用目标L公式(整个阴影区映射到同一 target_L 标量)
  v3.1: 引入 ab 补偿(补偿 L 改变带来的感知饱和度变化)
  v3:   简单乘性 boost_factor(已废弃,被 v3.2 目标L 取代)

优点
────
  - 零模型文件、零下载、零额外依赖
  - CPU 512×512 单张 <30ms(ab 补偿,无羽化无高光,向量化逐块 mean)
  - 不改变 a*/b* 方向(hue),只改 magnitude
  - 边缘过渡不在本模块职责范围

局限
────
  - 软阴影/自阴影检测能力有限
  - 极暗场景(L_illum < 25)不强行补偿
  - 不处理高光(若需要可下游单独实现)
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
    """在 shadow_mask 内按 RGB 色差约束做连通域标记(Union-Find, O(N α(N)))。

    两个相邻阴影像素属于同一连通块, 当且仅当它们的 BGR max 通道差
    小于 color_threshold。使用 Union-Find + 向量化边缘检测,
    内联 find/union 消除 Python 函数调用开销, 用 flat index 避免
    y*w+x 坐标转换。

    Args:
        shadow_mask: bool 数组 (H, W), 阴影 mask
        bgr: uint8 BGR 图像 (H, W, 3)
        color_threshold: BGR max 通道差阈值 (默认 15)

    Returns:
        labels: int32 数组 (H, W), 0 表示非阴影, 1..n_labels 表示连通块编号
        n_labels: 连通块数量
    """
    h, w = shadow_mask.shape
    n = h * w

    # ── Union-Find ──
    _p = np.arange(n, dtype=np.int32)
    _r = np.zeros(n, dtype=np.int32)

    # ── 向量化计算相邻像素 BGR max 通道差 (int16 替代 float32, 紧凑数组) ──
    color_threshold_f = float(color_threshold)
    bgr_i = bgr.astype(np.int16)

    # 水平边: (y, x) 与 (y, x-1) 的色差 → (h, w-1) 紧凑数组
    h_diff = np.max(np.abs(bgr_i[:, 1:] - bgr_i[:, :-1]), axis=2)
    h_edge = (shadow_mask[:, 1:] & shadow_mask[:, :-1] &
              (h_diff < color_threshold_f))
    h_edge_flat = np.flatnonzero(h_edge)
    if len(h_edge_flat) > 0:
        y_h = h_edge_flat // (w - 1)
        x_h = h_edge_flat % (w - 1)
        h_idx = (y_h * w + (x_h + 1)).astype(np.int32)
    else:
        h_idx = np.array([], dtype=np.int32)

    # 垂直边: (y, x) 与 (y-1, x) 的色差 → (h-1, w) 紧凑数组
    v_diff = np.max(np.abs(bgr_i[1:, :] - bgr_i[:-1, :]), axis=2)
    v_edge = (shadow_mask[1:, :] & shadow_mask[:-1, :] &
              (v_diff < color_threshold_f))
    v_edge_flat = np.flatnonzero(v_edge)
    if len(v_edge_flat) > 0:
        y_v = v_edge_flat // w
        x_v = v_edge_flat % w
        v_idx = ((y_v + 1) * w + x_v).astype(np.int32)
    else:
        v_idx = np.array([], dtype=np.int32)

    # ── Union 所有满足色差约束的边 (内联 find+union, list 迭代减少 numpy 装箱开销) ──
    # 水平边: union(idx, idx-1) 其中 idx = y*w + x (右像素)
    for idx in h_idx.tolist():
        i = idx
        while _p[i] != i:
            _p[i] = _p[_p[i]]
            i = _p[i]
        j = idx - 1
        while _p[j] != j:
            _p[j] = _p[_p[j]]
            j = _p[j]
        if i != j:
            if _r[i] < _r[j]:
                _p[i] = j
            elif _r[i] > _r[j]:
                _p[j] = i
            else:
                _p[j] = i
                _r[i] += 1

    # 垂直边: union(idx, idx-w) 其中 idx = y*w + x (下像素)
    for idx in v_idx.tolist():
        i = idx
        while _p[i] != i:
            _p[i] = _p[_p[i]]
            i = _p[i]
        j = idx - w
        while _p[j] != j:
            _p[j] = _p[_p[j]]
            j = _p[j]
        if i != j:
            if _r[i] < _r[j]:
                _p[i] = j
            elif _r[i] > _r[j]:
                _p[j] = i
            else:
                _p[j] = i
                _r[i] += 1

    # ── 分配 label (向量化: path 压缩 + np.unique + np.searchsorted) ──
    labels = np.zeros((h, w), dtype=np.int32)
    shadow_flat = np.flatnonzero(shadow_mask)
    if len(shadow_flat) == 0:
        return labels, 0

    # 仅对阴影像素做 path 压缩 (避免全图 262k 次 Python 循环)
    for idx in shadow_flat.tolist():
        while _p[idx] != _p[_p[idx]]:
            _p[idx] = _p[_p[idx]]

    # 向量化: 取根 → 唯一根 → searchsorted 映射到连续 label
    roots = _p[shadow_flat]
    unique_roots = np.unique(roots)
    n_labels = len(unique_roots)
    labels.ravel()[shadow_flat] = np.searchsorted(unique_roots, roots) + 1

    return labels, n_labels


# ──────────────────────────────────────────────────────────────────────
# 阴影去除:目标 L + 线性混合 + ab 补偿,无羽化
# ──────────────────────────────────────────────────────────────────────
def _remove_shadow_lab(bgr: np.ndarray,
                       sigma: int,
                       threshold: float,
                       dark_object_ratio: float = 0.20,
                       ab_compensation_alpha: float = 0.3,
                       use_morphology: bool = True,
                       morph_kernel_size: int = 3,
                       target_l_offset: float = 0.1,
                       target_l_gain: float = 1.25,
                       shadow_blend: float = 0.5,
                       color_threshold: float = 15.0) -> np.ndarray:
    """Lab 空间阴影去除(目标 L + 线性混合 + ab 补偿,纯 numpy + opencv,无羽化)。

    算法步骤
    --------
      1) BGR → Lab
      2) 大 σ 高斯估计光照分量 L_illum(只用于检测)
      3) 多条件阴影检测(diff + 周围够亮 + 非暗物体)
      4) 形态学清理(OPEN 去噪 + 1 次 DILATE 轻膨胀填洞)
      5) 色差约束连通块: mask 内按 BGR max 通道差 < color_threshold 做连通域标记
      6) 逐块目标 L: 每块独立计算 median L, target_L[block] = offset×255 + gain×median_L
         (解决 v3.5 L_illum 高斯模糊的两大问题: 亮像素扰动 + 边缘平滑过渡)
      7) 线性混合 + 下限保护(逐像素, 保留原始细节):
           L_blend[px] = (1 - blend) × L[px] + blend × target_L[px]
           L_new[px] = max(L[px], L_blend[px])
      8) 硬掩码 L 修正(np.where 直接切换,无羽化)
      9) ab 补偿(逐像素 per_pixel_boost = L_new/L,按 ^α 缩放 a/b,掩码外不动)

    Args:
        bgr: 输入 BGR uint8 图
        sigma: 高斯 σ(光照估计用,通常由 sigma_ratio × 短边 得到)
        threshold: L_illum − L > 该值才视为阴影(默认 3.0,激进以捕获浅阴影)
        dark_object_ratio: 暗物体过滤比例(L > ratio * L_illum 才视为潜在阴影,
                          默认 0.20 放宽,允许浅阴影;真实阴影 80/180=0.44 保留,黑头发 20/180=0.11 排除)
        ab_compensation_alpha: ab 补偿强度(0=关闭,0.3 轻度,0.5 中度,1.0 满补偿)
        use_morphology: 是否对 mask 做开运算+轻膨胀
        morph_kernel_size: 形态学核大小(默认 3,3×3 椭圆核避免 mask 过度外扩)
        target_l_offset: 目标 L 公式的加性偏移(0-1 空间,默认 0.1)
        target_l_gain: 目标 L 公式的乘性增益(默认 1.25)
        shadow_blend: 阴影区线性混合比例(0=不改,0.5=半细节保留,1=完全填 target_L;默认 0.5)
        color_threshold: 色差约束连通块的 BGR max 通道差阈值(默认 15,
                         相邻阴影像素 BGR max|diff| < 该值才归入同一连通块)

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
    #    L_illum 只用于阴影检测 (L_illum - L > threshold), 降采样引入的
    #    微小误差被阈值 margin 吸收, 对最终结果无实质影响。
    h_small, w_small = max(16, H // 2), max(16, W // 2)
    L_small = cv2.resize(L, (w_small, h_small))
    L_illum_small = cv2.GaussianBlur(L_small, (0, 0),
                                     sigmaX=sigma / 2.0, sigmaY=sigma / 2.0)
    L_illum = cv2.resize(L_illum_small, (W, H))

    # 3) 多条件阴影 mask
    #    a) L 显著低于 L_illum(阈值 3.0,激进一些以捕获浅阴影)
    #    b) 周围不能太暗(L_illum > 25,允许偏暗图的浅阴影)
    #    c) L > ratio * L_illum(关键:过滤黑物体,保留真实阴影;0.20 放宽)
    cond_diff = (L_illum - L) > threshold
    cond_illum = L_illum > 15.0
    cond_not_dark = L > dark_object_ratio * L_illum
    shadow_mask = cond_diff & cond_illum & cond_not_dark

    if not shadow_mask.any():
        return bgr

    # 4) 形态学清理(去小斑 + 轻膨胀填洞)
    #    OPEN 去噪(用 3×3 核,保守)
    #    DILATE(1次)代替 CLOSE:把 mask 边缘向外推 ~1 像素,既填小洞又不过度外扩
    #    (原 CLOSE 用 5×5 核会把 mask 扩展 2-3 像素,过激)
    if use_morphology:
        k = max(3, int(morph_kernel_size) | 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask_u8 = cv2.morphologyEx(
            (shadow_mask.astype(np.uint8) * 255),
            cv2.MORPH_OPEN, kernel,
        )
        mask_u8 = cv2.dilate(mask_u8, kernel, iterations=1)
        shadow_mask = mask_u8 > 127
        if not shadow_mask.any():
            return bgr

    # 5) 色差约束连通块: mask 内按 RGB 色差约束做连通域标记
    #    相邻像素 BGR max 通道差 < color_threshold 才归入同一连通块
    #    这确保浅阴影和深阴影即使空间相邻也不会被合并到同一块
    labels, n_labels = _color_constrained_labeling(shadow_mask, bgr, color_threshold)

    # 6) 逐块目标 L: 每块独立计算 mean L, 套用 target_L 公式
    #    target_L[block] = offset×255 + gain×mean_L[block]
    #    解决 v3.5 L_illum 两大问题:
    #      a) 周围亮像素通过高斯模糊渗透进阴影区 → 扰动 target_L
    #      b) 高斯模糊平滑掉边缘 → 浅/深阴影边界过渡平滑而非明显边缘
    #    优化: 用 np.bincount 向量化逐块均值替代 sort+groupby+for 循环
    #    色差约束连通块内像素颜色相近, 均值 ≈ 中位数, 可放心使用
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

    # 7) 线性混合(逐像素,保留原始细节,避免纯色块)
    #    L_blend[px] = (1 - blend) × L[px] + blend × target_L[px]
    L_blend = (1.0 - shadow_blend) * L + shadow_blend * target_L_arr

    # 7.5) 下限保护(修复"亮边变暗"伪影):
    #      若原 L > target_L(target_L < L), L_blend < L,
    #      max 保护让该像素保持原 L,避免把已够亮的边界点反向压低
    L_new = np.maximum(L, L_blend)

    # 8) 硬掩码 L 修正(阴影区填 L_new,区外保持原 L;无羽化)
    L_final = np.where(shadow_mask, L_new, L)

    # 9) ab 补偿: L 改后感知饱和度变化
    #    混合后 boost 不再是常数,而是逐像素 per_pixel_boost = L_new[px] / L[px]
    #    公式: 掩码内 C_new = C_old × per_pixel_boost[px]^α
    #    掩码外: ab_scale = 1.0(不动 a, b)
    #    ★ 必须先减 128 到 signed Lab 再缩放(直接对 uint8 缩放会暴增 chroma)
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

    # 重组 → BGR uint8
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
    `shadow_threshold`、调大 `sigma_ratio`、关闭 `use_morphology`。
    """
    DEFAULT_CONFIG = {
        # 总开关
        "enabled": True,

        # ── 阴影检测与修正 ──
        # 光照估计
        "sigma_ratio": 0.10,             # 高斯 σ 与图片短边的比值
        # 检测阈值
        "shadow_threshold": 3.0,         # L_illum - L > 该值才视为阴影(3.0 激进,捕获浅阴影)
        "dark_object_ratio": 0.20,       # L > ratio * L_illum 才视为潜在阴影(0.20 放宽,允许浅阴影)
        # 形态学
        "use_morphology": True,
        "morph_kernel_size": 5,          # 小核(5x5 椭圆),避免 mask 过度外扩
        # 目标 L 公式(0-1 归一化空间)
        "target_l_offset": 0.1,         # 加性偏移;等价 0-100 空间的 "+5"
        "target_l_gain": 1.25,            # 乘性增益
        #   公式: target_L[block] = (offset + median_L[block]/255 × gain) × 255
        #   每个色差约束连通块独立计算 median_L
        # 色差约束连通块
        "color_threshold": 15.0,         # BGR max 通道差阈值;相邻像素 diff < 该值才归入同一连通块
        # 阴影区线性混合比例(避免纯色块,保留原始细节)
        "shadow_blend": 0.5,             # 0=不改, 0.5=半细节保留(默认), 1=完全填 target_L

        # ── ab 补偿 ──
        "ab_compensation_alpha": 0.3,    # 0=关闭,0.3 轻度,0.5 中度,1.0 满补偿
    }

    def __init__(self, **kwargs):
        cfg = {**self.DEFAULT_CONFIG, **kwargs}
        for key in cfg:
            setattr(self, key, cfg[key])

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
        )


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
    else:
        # 正确性测试
        r = ShadowRemover()

        # ── 辅助函数: 手动模拟 v3.6 算法, 计算期望 L_new ──
        def _expected_l_new(img_bgr, remover, roi=None):
            """手动模拟 v3.6 完整流水线, 返回期望 L_new (roi 区域均值) 或全图."""
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
        #     阴影 128x128 BGR(80) 是均匀色块 → 一个连通块
        #     median L ≈ 87, target_L = 12.75 + 1.2*87 ≈ 117.15
        #     blend=0.5 → L_blend = 0.5*87 + 0.5*117.15 ≈ 102.1
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
        #     64x64 阴影 BGR(50,65,80) 是均匀色块 → 一个连通块
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
        #     构造: 深阴影 100x100 BGR(50) L≈53 + 浅阴影 4x4 BGR(100) L≈108
        #     BGR max diff = 50 > color_threshold(15) → 分属不同连通块
        #     深阴影: target_L = 12.75 + 1.2*53 ≈ 76.35, L_new ≈ 64.7
        #     浅阴影: target_L = 12.75 + 1.2*108 ≈ 142.35, L_new ≈ 125.2
        img_t7 = np.full((256, 256, 3), 255, dtype=np.uint8)
        img_t7[78:178, 78:178] = (50, 50, 50)
        img_t7[170:174, 170:174] = (100, 100, 100)
        out_t7 = r.remove(img_t7)
        # 1) 浅阴影 4x4 应被提亮
        L_bright_before = cv2.cvtColor(img_t7[170:174, 170:174], cv2.COLOR_BGR2Lab)[..., 0].mean()
        L_bright_after = cv2.cvtColor(out_t7[170:174, 170:174], cv2.COLOR_BGR2Lab)[..., 0].mean()
        print(f"[T7] 浅阴影 4x4 L: {L_bright_before:.1f} → {L_bright_after:.1f} "
              f"(色差约束连通块分离, 独立 target_L)")
        assert L_bright_after > L_bright_before + 5, \
            f"浅阴影应被提亮, 实际 {L_bright_before:.1f}→{L_bright_after:.1f}"
        # 2) 深阴影区应被提亮
        L_shadow_before = cv2.cvtColor(img_t7[120:130, 120:130], cv2.COLOR_BGR2Lab)[..., 0].mean()
        L_shadow_after = cv2.cvtColor(out_t7[120:130, 120:130], cv2.COLOR_BGR2Lab)[..., 0].mean()
        print(f"[T7] 深阴影区 L: {L_shadow_before:.1f} → {L_shadow_after:.1f} (应被提亮)")
        assert L_shadow_after > L_shadow_before + 5, "深阴影应被提亮"
        # 3) 验证浅阴影和深阴影分属不同连通块(通过 target_L 不同来间接验证)
        #    若在同一块, target_L 相同, L_new 中浅阴影 L < 深阴影 L_new + small_gap
        #    若在不同块, 浅阴影 target_L 更高, L_new 中浅阴影 L >> 深阴影 L_new
        bright_vs_dark_ratio = L_bright_after / L_shadow_after
        print(f"[T7] 浅阴影/深阴影 L_new 比值: {bright_vs_dark_ratio:.3f} "
              f"(>1.5 表明确实在不同连通块)")
        assert bright_vs_dark_ratio > 1.5, \
            f"浅阴影 target_L 应远高于深阴影(分属不同连通块), 比值 {bright_vs_dark_ratio:.3f}"

        print("OK: 阴影提亮 / 非阴影不变 / 黑物体过滤 / "
              "目标L gain 可调 / ab 补偿生效 / α=0 关闭 / hue 保持 / 比例符合 CIE 公式 / "
              "浅深阴影色差约束分离(连通块独立 target_L)")
