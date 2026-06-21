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
    args = parser.parse_args()

    total_episodes = args.episodes
    patience = args.patience
    use_shadow_removal = not args.no_shadow_removal

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    print(f"阴影去除: {'关闭' if not use_shadow_removal else '开启'}")

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
                             use_shadow_removal=use_shadow_removal)
    val_set = ColorDataset(img_dir, target_jsons, 'val', pixiv_download_dir=DEFAULT_PIXIV_DIR,
                           use_shadow_removal=use_shadow_removal)

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
            mask_mean = mask.mean(dim=[1, 2, 3])
            loss_mask = ((mask_mean - 0.4) ** 2).mean()
            loss = loss_col + 0.1 * loss_mask
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
                val_loss += (mse(pred_fg, lab_fg) + mse(pred_bg, lab_bg)).item() * img.size(0)
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
