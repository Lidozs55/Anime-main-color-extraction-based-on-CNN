import sys, os, glob, argparse
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader

# 添加父目录到路径以便导入 model 和 dataset
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import ColorNetMasked
from dataset import ColorDataset

CHECKPOINT_FILE = 'checkpoint.pth'
BEST_MODEL_FILE = 'best_model.pth'


def find_all_targets(root_dir='..'):
    """查找所有 targets*.json 文件"""
    targets = []
    tj = os.path.join(root_dir, 'targets.json')
    if os.path.exists(tj):
        targets.append(tj)
    pattern = os.path.join(root_dir, 'targets_pixiv_*.json')
    pixiv_targets = sorted(glob.glob(pattern))
    targets.extend(pixiv_targets)
    return targets


def main():
    parser = argparse.ArgumentParser(description="训练 ColorNet-Masked 学生模型")
    parser.add_argument('--episodes', type=int, default=200, help='训练总 epoch 数（默认 200）')
    parser.add_argument('--batch-size', type=int, default=32, help='mini-batch size（梯度累积用，默认 32）')
    parser.add_argument('--patience', type=int, default=30, help='early stopping patience（默认 30）')
    parser.add_argument('--lr', type=float, default=1e-3, help='学习率（默认 1e-3）')
    parser.add_argument('--ema-alpha', type=float, default=0.1, help='EMA 平滑系数（默认 0.1）')
    args = parser.parse_args()

    total_episodes = args.episodes
    patience = args.patience
    ema_alpha = args.ema_alpha

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 查找所有 targets JSON 文件
    target_jsons = find_all_targets()
    if not target_jsons:
        print("错误: 未找到任何 targets*.json 文件")
        sys.exit(1)

    print(f"找到 {len(target_jsons)} 个目标文件:")
    for tj in target_jsons:
        print(f"  - {tj}")

    img_dir = os.path.join('..', 'extracted_imgs', 'imgs')
    train_set = ColorDataset(img_dir, target_jsons, 'train')
    val_set = ColorDataset(img_dir, target_jsons, 'val')
    n_train = len(train_set)
    n_val = len(val_set)

    print(f"\n训练集: {n_train} 张  验证集: {n_val} 张")
    if n_train == 0:
        print("错误: 没有可用的训练数据")
        sys.exit(1)
    if n_val == 0:
        print("错误: 验证集为空")
        sys.exit(1)

    # 计算梯度累积步数：每个 mini-batch 后累积梯度，遍历全集后统一 update
    # 这确保每次参数更新都基于全体训练样本的梯度，消除 batch 分布偏差
    effective_batch = min(args.batch_size, n_train)
    accum_steps = max(1, (n_train + effective_batch - 1) // effective_batch)
    actual_batch = (n_train + accum_steps - 1) // accum_steps  # 均匀分配
    print(f"梯度累积: {accum_steps} 步 × {actual_batch} 张/步 = 全集 {n_train} 张")

    num_workers = min(4, max(0, os.cpu_count() or 1))
    train_loader = DataLoader(train_set, batch_size=actual_batch, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=max(n_val, 1), shuffle=False, num_workers=num_workers)

    model = ColorNetMasked().to(device)
    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    mse = nn.MSELoss()
    best_loss = float('inf')
    best_epoch = 0
    start_epoch = 0
    no_improve = 0
    best_model_state = None  # 内存中维护最佳模型权重
    ema_loss = None  # EMA 平滑 train_loss

    # 加载检查点
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_episodes)
    if os.path.exists(CHECKPOINT_FILE):
        ckpt = torch.load(CHECKPOINT_FILE, map_location=device, weights_only=False)
        best_loss = ckpt['best_loss']
        best_epoch = ckpt.get('best_epoch', 0)
        ema_loss = ckpt.get('ema_loss', None)

        # 恢复策略：从最佳模型权重重新开始，而非从最后（可能过拟合的）epoch 继续
        if 'best_model' in ckpt:
            model.load_state_dict(ckpt['best_model'])
            best_model_state = {k: v.clone() for k, v in ckpt['best_model'].items()}
            print(f"\n>>> 从最佳模型恢复: best_epoch={best_epoch}, best_loss={best_loss:.4f}")
        else:
            # 兼容旧版 checkpoint（无 best_model 字段）
            model.load_state_dict(ckpt['model'])
            best_model_state = {k: v.clone() for k, v in ckpt['model'].items()}
            print(f"\n>>> 从检查点恢复（旧格式）: epoch={ckpt['epoch']}, best_loss={best_loss:.4f}")
        # 重置 optimizer 和 scheduler（从最佳模型重新优化）
        opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_episodes)
        start_epoch = best_epoch
        print(f">>> 从第 {start_epoch} 轮重新开始训练（no_improve 重置为 0），目标 {total_episodes} epochs")
        no_improve = 0
    else:
        print(f"\n未找到检查点，从头开始训练（目标 {total_episodes} epochs）")

    for epoch in range(start_epoch, total_episodes):
        # ── 训练：梯度累积覆盖全集 ──
        model.train()
        epoch_loss = 0
        opt.zero_grad()

        for step, (img, lab_fg, lab_bg, conf) in enumerate(train_loader):
            img, lab_fg, lab_bg, conf = img.to(device), lab_fg.to(device), lab_bg.to(device), conf.to(device)
            pred_fg, pred_bg, mask = model(img)
            # confidence-weighted color loss
            loss_fg = (conf[:, 0] * ((pred_fg - lab_fg) ** 2).mean(dim=1)).mean()
            loss_bg = (conf[:, 1] * ((pred_bg - lab_bg) ** 2).mean(dim=1)).mean()
            loss_col = loss_fg + loss_bg
            mask_mean = mask.mean(dim=[1, 2, 3])
            loss_mask = ((mask_mean - 0.4) ** 2).mean()
            loss = (loss_col + 0.1 * loss_mask) / accum_steps
            loss.backward()
            epoch_loss += loss.item() * img.size(0) * accum_steps

        # 梯度累积完成，全集梯度已就绪，统一 update
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        opt.step()
        sched.step()

        train_loss = epoch_loss / n_train

        # ── 验证：检测过拟合的真实信号 ──
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for img, lab_fg, lab_bg, _ in val_loader:
                img, lab_fg, lab_bg = img.to(device), lab_fg.to(device), lab_bg.to(device)
                pred_fg, pred_bg, _ = model(img)
                val_loss += (mse(pred_fg, lab_fg) + mse(pred_bg, lab_bg)).item() * img.size(0)
        val_loss /= n_val

        # EMA 平滑 train_loss（辅助监控）
        if ema_loss is None:
            ema_loss = train_loss
        else:
            ema_loss = ema_alpha * train_loss + (1 - ema_alpha) * ema_loss

        print(f'Epoch {epoch+1:02d}/{total_episodes:02d}  '
              f'train={train_loss:.4f}  val={val_loss:.4f}  ema={ema_loss:.4f}')

        # 模型选择：基于 val_loss（真实泛化信号）
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch + 1
            no_improve = 0
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            torch.save({
                'epoch': best_epoch,
                'model': model.state_dict(),
                'best_loss': best_loss,
            }, BEST_MODEL_FILE)
            print(f'  -> 保存最佳模型 (val_loss={val_loss:.4f}, 第 {best_epoch} 轮)')
        else:
            no_improve += 1

        # 保存训练检查点（用于断点续传）
        torch.save({
            'epoch': epoch + 1,
            'model': model.state_dict(),
            'optimizer': opt.state_dict(),
            'scheduler': sched.state_dict(),
            'best_model': best_model_state,
            'best_loss': best_loss,
            'best_epoch': best_epoch,
            'no_improve': no_improve,
            'ema_loss': ema_loss,
            'total_episodes': total_episodes,
        }, CHECKPOINT_FILE)

        # Early stopping
        if no_improve >= patience:
            print(f'\nEarly stopping at epoch {epoch+1} '
                  f'(no improvement for {patience} epochs, best at epoch {best_epoch})')
            break

    print('\nTraining done.')
    print(f'最佳 val_loss: {best_loss:.4f} (出现在第 {best_epoch} 轮)')
    print(f'最佳模型已保存至: {BEST_MODEL_FILE}')


if __name__ == '__main__':
    main()
