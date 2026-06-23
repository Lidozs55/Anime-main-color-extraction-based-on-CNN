"""
学生模型训练脚本。

用法:
    cd student
    python train.py --episodes 60 --patience 15
    python train.py --episodes 100 --batch-size 32 --lr 5e-4

数据流程:
    1) find_all_targets() 扫描项目根目录所有 targets*.json
    2) ColorDataset 同时读取本地 img/ 与 pixiv_img/ 中的图片
    3) 8:2 划分 train/val,训练集开启数据增强(翻转/旋转/随机裁剪)
    4) AdamW + CosineAnnealingLR,val_loss 连续 --patience 轮不下降则早停
    5) best_model.pth     仅 weights,val_loss 最佳时覆盖
       checkpoint.pth     含 optimizer/scheduler/epoch,断点续训用
"""
import sys, os, glob, argparse
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader

# 添加父目录到路径以便导入 model 和 dataset
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import ColorNetMasked
from dataset import ColorDataset

CHECKPOINT_FILE = 'checkpoint.pth'
BEST_MODEL_FILE = 'best_model.pth'

# 项目根目录(此脚本位于 student/ 下,根目录为上一级)
PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
DEFAULT_IMG_DIR = os.path.join(PROJECT_ROOT, 'img')
DEFAULT_PIXIV_DIR = os.path.join(PROJECT_ROOT, 'pixiv_img')


# ──────────────────────────────────────────────────────────────────────
# sRGB 色域 LUT: C*_max(L, h)
#
# 对 sRGB 立方体 (256³) 遍历,反投影到 CIELAB,在 (L, h) 网格上取最大色度 C*。
# 训练时按 (L_target, h_target) 查表得到当前色系的感知"最大可及色度",
# 归一化色度 s = C_target / C*_max 作为饱和度权重,比简单 |ab|/128 更均匀。
#
# 预计算 ~0.8s,生成 201×360 float32 ≈ 283 KB,首次调用时构建并缓存。
# ──────────────────────────────────────────────────────────────────────
_CMAX_LUT = None


def _build_cmax_lut():
    """构建 sRGB 色域的 C*_max(L, h) LUT,返回 (201, 360) float32 tensor。"""
    import cv2
    N = 256
    R, G, B = np.meshgrid(np.arange(N, dtype=np.uint8),
                           np.arange(N, dtype=np.uint8),
                           np.arange(N, dtype=np.uint8),
                           indexing='ij')
    bgr = np.stack([B.ravel(), G.ravel(), R.ravel()], axis=1)
    # float32 输入,得到 [0,100] L 范围;OpenCV Lab a/b∈[0,255] 对应 [-128,127]
    bgr_f = (bgr.astype(np.float32) / 255.0).reshape(1, -1, 3)
    lab_flat = cv2.cvtColor(bgr_f, cv2.COLOR_BGR2Lab).reshape(-1, 3)
    L = lab_flat[:, 0]
    a = lab_flat[:, 1] - 128.0
    b = lab_flat[:, 2] - 128.0
    C = np.sqrt(a * a + b * b).astype(np.float32)
    h = ((np.degrees(np.arctan2(b, a)) + 360.0) % 360.0).astype(np.float32)

    L_BINS, H_BINS = 201, 360
    L_idx = np.clip(np.round(L * 2.0).astype(np.int32), 0, L_BINS - 1)
    H_idx = np.floor(h).astype(np.int32) % H_BINS

    # 按 (H, L) 排序后分组取最大
    keys = H_idx * L_BINS + L_idx
    order = np.argsort(keys)
    keys_s, C_s = keys[order], C[order]
    unique_keys, first_idx = np.unique(keys_s, return_index=True)
    splits = np.split(C_s, first_idx[1:])
    Cmax = np.zeros((L_BINS, H_BINS), dtype=np.float32)
    for k, c in zip(unique_keys, splits):
        hh, ll = k // L_BINS, k % L_BINS
        Cmax[ll, hh] = c.max() if len(c) > 0 else 0.0
    return torch.from_numpy(Cmax)


def _get_cmax_lut() -> torch.Tensor:
    """惰性构建/获取 C*_max(L, h) LUT,返回 (201, 360) float32。"""
    global _CMAX_LUT
    if _CMAX_LUT is None:
        _CMAX_LUT = _build_cmax_lut()
    return _CMAX_LUT


def _ab_cosine_penalty(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """AB余弦惩罚: L*(100-L)/100 × s × (1 - cos⟨(a₁,b₁),(a₂,b₂)⟩)

    其中:
      - L 取 target_L(L∈[0,100])
      - s = C_target / C*_max(L_target, h_target),归一化色度,感知更均匀
      - 1-cos 惩罚 (a,b) 向量方向差异(hue)

    性能:权重计算在 no_grad 下完成,autograd graph 仅保留
    ``weight * (1 - cos)`` 一个乘法 + ``1 - cos`` 一段子图,显著加速反向。
    """
    a1, b1 = pred[:, 1], pred[:, 2]
    a2, b2 = target[:, 1], target[:, 2]

    # ── 权重:全部在 no_grad 下完成,避免 autograd 跟踪 ──
    with torch.no_grad():
        L_t = target[:, 0].clamp(0.0, 100.0)                # (B,)
        C_target = torch.sqrt(a2 * a2 + b2 * b2)
        h_target = (torch.atan2(b2, a2) * (180.0 / np.pi) + 360.0) % 360.0
        lut = _get_cmax_lut().to(pred.device)               # (201, 360)
        L_idx_f = (L_t * 2.0).clamp(0, 200)
        h_idx_f = h_target.clamp(0.0, 359.999)
        l0 = L_idx_f.floor().long().clamp(0, 200)
        l1 = (l0 + 1).clamp(0, 200)
        h0 = h_idx_f.floor().long().clamp(0, 359)
        h1 = (h0 + 1).clamp(0, 359)
        wl = L_idx_f - l0.float()
        wh = h_idx_f - h0.float()
        Cmax00 = lut[l0, h0]; Cmax01 = lut[l0, h1]
        Cmax10 = lut[l1, h0]; Cmax11 = lut[l1, h1]
        Cmax_t = (Cmax00 * (1 - wl) * (1 - wh)
                  + Cmax01 * (1 - wl) * wh
                  + Cmax10 * wl * (1 - wh)
                  + Cmax11 * wl * wh).clamp(min=1.0)
        sat_norm = (C_target / Cmax_t).clamp(0.0, 1.0)
        weight = (L_t * (100.0 - L_t) / 10.0) * sat_norm    # (B,)

    # ── 1 - cos 部分参与 autograd,梯度回传到 pred ──
    # 用 |a1|² × |a2|² 一次 rsqrt,避免两次 sqrt
    inv_prod = torch.rsqrt((a1 * a1 + b1 * b1) * (a2 * a2 + b2 * b2) + 1e-8)
    one_minus_cos = 1.0 - (a1 * a2 + b1 * b2) * inv_prod

    return (weight * one_minus_cos).mean()


def _ab_cosine_penalty_pair(pred_fg: torch.Tensor, pred_bg: torch.Tensor,
                              tgt_fg: torch.Tensor, tgt_bg: torch.Tensor) -> torch.Tensor:
    """批量计算 fg+bg 的 ab_cos,合并两次调用的 autograd 开销。

    权重(基于 target)仍在 no_grad 下完成;1-cos 部分共享调用栈,
    减少 autograd graph 重建次数。
    """
    a1f, b1f = pred_fg[:, 1], pred_fg[:, 2]
    a1b, b1b = pred_bg[:, 1], pred_bg[:, 2]
    a2f, b2f = tgt_fg[:, 1], tgt_fg[:, 2]
    a2b, b2b = tgt_bg[:, 1], tgt_bg[:, 2]

    with torch.no_grad():
        # fg
        L_f = tgt_fg[:, 0].clamp(0.0, 100.0)
        C_f = torch.sqrt(a2f * a2f + b2f * b2f)
        h_f = (torch.atan2(b2f, a2f) * (180.0 / np.pi) + 360.0) % 360.0
        # bg
        L_b = tgt_bg[:, 0].clamp(0.0, 100.0)
        C_b = torch.sqrt(a2b * a2b + b2b * b2b)
        h_b = (torch.atan2(b2b, a2b) * (180.0 / np.pi) + 360.0) % 360.0

        lut = _get_cmax_lut().to(pred_fg.device)

        def _norm_sat(L_t, C_t, h_t):
            L_idx_f = (L_t * 2.0).clamp(0, 200)
            h_idx_f = h_t.clamp(0.0, 359.999)
            l0 = L_idx_f.floor().long().clamp(0, 200)
            l1 = (l0 + 1).clamp(0, 200)
            h0 = h_idx_f.floor().long().clamp(0, 359)
            h1 = (h0 + 1).clamp(0, 359)
            wl = L_idx_f - l0.float()
            wh = h_idx_f - h0.float()
            Cmax = (lut[l0, h0] * (1 - wl) * (1 - wh)
                    + lut[l0, h1] * (1 - wl) * wh
                    + lut[l1, h0] * wl * (1 - wh)
                    + lut[l1, h1] * wl * wh).clamp(min=1.0)
            return (L_t * (100.0 - L_t) / 100.0) * (C_t / Cmax).clamp(0.0, 1.0)

        w_f = _norm_sat(L_f, C_f, h_f)
        w_b = _norm_sat(L_b, C_b, h_b)

    # 1 - cos 仍参与 autograd
    inv_f = torch.rsqrt((a1f * a1f + b1f * b1f) * (a2f * a2f + b2f * b2f) + 1e-8)
    inv_b = torch.rsqrt((a1b * a1b + b1b * b1b) * (a2b * a2b + b2b * b2b) + 1e-8)
    omc_f = 1.0 - (a1f * a2f + b1f * b2f) * inv_f
    omc_b = 1.0 - (a1b * a2b + b1b * b2b) * inv_b
    return (w_f * omc_f).mean() + (w_b * omc_b).mean()


def find_all_targets(root_dir=None):
    """查找所有 targets*.json 文件(默认在项目根目录)。

    顺序:targets.json 优先,然后按文件名排序的 targets_pixiv_*.json / targets_*.json。
    这样保证多个 JSON 之间的 key 唯一性,可被 ColorDataset 合并加载。
    """
    if root_dir is None:
        root_dir = PROJECT_ROOT
    targets = []
    tj = os.path.join(root_dir, 'targets.json')
    if os.path.exists(tj):
        targets.append(tj)
    # 也兼容 label.py 输出的 targets_*.json
    for pat in ('targets_pixiv_*.json', 'targets_*.json'):
        for f in sorted(glob.glob(os.path.join(root_dir, pat))):
            if f not in targets:
                targets.append(f)
    return targets


def main():
    parser = argparse.ArgumentParser(description="训练 ColorNet-Masked 学生模型")
    parser.add_argument('--episodes', type=int, default=60, help='训练总 epoch 数（默认 60）')
    parser.add_argument('--batch-size', type=int, default=64, help='batch size（默认 64）')
    parser.add_argument('--patience', type=int, default=15, help='early stopping patience（默认 15）')
    parser.add_argument('--lr', type=float, default=1e-3, help='学习率（默认 1e-3）')
    parser.add_argument('--no-shadow-removal', action='store_true',
                        help='关闭阴影去除(默认开启,与 preview.py 推理时一致)')
    parser.add_argument('--cache-dir', type=str, default=None,
                        help='阴影去除缓存目录(默认: student/.shadow_cache/);'
                             '与 --no-shadow-removal 互斥')
    parser.add_argument('--no-cache', action='store_true',
                        help='禁用阴影去除缓存(每个 epoch 重新计算)')
    args = parser.parse_args()

    total_episodes = args.episodes
    patience = args.patience
    use_shadow_removal = not args.no_shadow_removal
    # 缓存目录:--no-cache 禁用;否则用 --cache-dir 或默认
    cache_dir = None if args.no_cache else args.cache_dir

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    print(f"阴影去除: {'关闭' if not use_shadow_removal else '开启'}")
    if use_shadow_removal:
        print(f"阴影缓存: {'禁用' if cache_dir is None else cache_dir}")

    # 查找所有 targets JSON 文件
    target_jsons = find_all_targets()
    if not target_jsons:
        print("错误: 未找到任何 targets*.json 文件")
        sys.exit(1)

    print(f"找到 {len(target_jsons)} 个目标文件:")
    for tj in target_jsons:
        print(f"  - {tj}")

    img_dir = DEFAULT_IMG_DIR
    train_set = ColorDataset(img_dir, target_jsons, 'train', pixiv_download_dir=DEFAULT_PIXIV_DIR,
                             use_shadow_removal=use_shadow_removal, cache_dir=cache_dir)
    val_set = ColorDataset(img_dir, target_jsons, 'val', pixiv_download_dir=DEFAULT_PIXIV_DIR,
                           use_shadow_removal=use_shadow_removal, cache_dir=cache_dir)

    print(f"\n训练集: {len(train_set)} 张")
    print(f"验证集: {len(val_set)} 张")
    total = len(train_set) + len(val_set)
    if total == 0:
        print("错误: 没有可用的训练数据")
        sys.exit(1)

    num_workers = min(4, max(0, os.cpu_count() or 1))
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=num_workers)

    model = ColorNetMasked().to(device)
    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    mse = nn.MSELoss()
    best_loss = float('inf')
    best_epoch = 0
    start_epoch = 0
    no_improve = 0

    # 加载检查点
    if os.path.exists(CHECKPOINT_FILE):
        ckpt = torch.load(CHECKPOINT_FILE, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model'])
        opt.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt['epoch']
        best_loss = ckpt['best_loss']
        best_epoch = ckpt.get('best_epoch', 0)
        no_improve = ckpt.get('no_improve', 0)

        # 初始化 scheduler（T_max 用总轮次），然后手动设置 last_epoch 恢复状态
        # 注意：先做一次 optimizer.step() 空操作以消除 PyTorch 的 step 顺序警告
        sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_episodes, last_epoch=start_epoch - 1)
        opt.step()  # dummy step to satisfy scheduler requirement
        opt.zero_grad()
        print(f"\n>>> 从检查点恢复: epoch={start_epoch}, best_loss={best_loss:.4f} (epoch {best_epoch}), no_improve={no_improve}")
        print(f">>> 本次目标训练至 {total_episodes} epochs")
    else:
        sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_episodes)
        print(f"\n未找到检查点，从头开始训练（目标 {total_episodes} epochs）")

    for epoch in range(start_epoch, total_episodes):
        model.train()
        train_loss = 0
        for img, lab_fg, lab_bg in train_loader:
            img, lab_fg, lab_bg = img.to(device), lab_fg.to(device), lab_bg.to(device)
            pred_fg, pred_bg, mask = model(img)
            loss_col = mse(pred_fg, lab_fg) + mse(pred_bg, lab_bg)
            loss_ab_cos = _ab_cosine_penalty_pair(pred_fg, pred_bg, lab_fg, lab_bg)
            mask_mean = mask.mean(dim=[1, 2, 3])
            loss_mask = ((mask_mean - 0.4) ** 2).mean()
            loss = loss_col + 0.1 * loss_mask + loss_ab_cos
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss += loss.item() * img.size(0)
        train_loss /= len(train_set)
        sched.step()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for img, lab_fg, lab_bg in val_loader:
                img, lab_fg, lab_bg = img.to(device), lab_fg.to(device), lab_bg.to(device)
                pred_fg, pred_bg, _ = model(img)
                val_loss += (mse(pred_fg, lab_fg) + mse(pred_bg, lab_bg)
                         + _ab_cosine_penalty_pair(pred_fg, pred_bg, lab_fg, lab_bg)).item() * img.size(0)
        val_loss /= len(val_set)

        print(f'Epoch {epoch+1:02d}/{total_episodes:02d}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}')

        # 保存最佳模型（仅保存 weights，覆盖时保证 best_loss 是真正的最佳）
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch + 1
            no_improve = 0
            torch.save({
                'epoch': best_epoch,
                'model': model.state_dict(),
                'best_loss': best_loss,
            }, BEST_MODEL_FILE)
            print(f'  -> 保存最佳模型 (val_loss={val_loss:.4f}, 第 {best_epoch} 轮)')
        else:
            no_improve += 1

        # 保存训练检查点（用于断点续传，epoch 表示"下一个要跑的轮次"）
        torch.save({
            'epoch': epoch + 1,
            'model': model.state_dict(),
            'optimizer': opt.state_dict(),
            'best_loss': best_loss,
            'best_epoch': best_epoch,
            'no_improve': no_improve,
            'total_episodes': total_episodes,
        }, CHECKPOINT_FILE)

        # Early stopping
        if no_improve >= patience:
            print(f'\nEarly stopping at epoch {epoch+1} (no improvement for {patience} epochs, best at epoch {best_epoch})')
            break

    print('\nTraining done.')
    print(f'最佳 val_loss: {best_loss:.4f} (出现在第 {best_epoch} 轮)')
    print(f'最佳模型已保存至: {BEST_MODEL_FILE}')


if __name__ == '__main__':
    main()
