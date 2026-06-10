import sys, os, glob
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader

# 添加父目录到路径以便导入 model 和 dataset
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import ColorNetMasked
from dataset import ColorDataset

CHECKPOINT_FILE = 'checkpoint.pth'


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

    print(f"\n训练集: {len(train_set)} 张")
    print(f"验证集: {len(val_set)} 张")
    total = len(train_set) + len(val_set)
    if total == 0:
        print("错误: 没有可用的训练数据")
        sys.exit(1)

    num_workers = min(4, max(0, os.cpu_count() or 1))
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=64, shuffle=False, num_workers=num_workers)

    model = ColorNetMasked().to(device)
    opt = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=60)
    mse = nn.MSELoss()
    best_loss = float('inf')
    start_epoch = 0
    patience = 15
    no_improve = 0

    # 加载检查点
    if os.path.exists(CHECKPOINT_FILE):
        ckpt = torch.load(CHECKPOINT_FILE, map_location=device)
        model.load_state_dict(ckpt['model'])
        opt.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt['epoch']
        best_loss = ckpt['best_loss']
        no_improve = ckpt.get('no_improve', 0)
        # 恢复 scheduler 状态
        for _ in range(start_epoch):
            sched.step()
        print(f"\n>>> 从检查点恢复: epoch={start_epoch}, best_loss={best_loss:.4f}, no_improve={no_improve}")
    else:
        print("\n未找到检查点，从头开始训练")

    for epoch in range(start_epoch, 60):
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

        print(f'Epoch {epoch+1:02d}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}')

        # 保存最佳模型
        if val_loss < best_loss:
            best_loss = val_loss
            no_improve = 0
            torch.save(model.state_dict(), 'best_model.pth')
            print(f'  -> 保存最佳模型 (val_loss={val_loss:.4f})')
        else:
            no_improve += 1

        # 保存检查点
        torch.save({
            'epoch': epoch + 1,
            'model': model.state_dict(),
            'optimizer': opt.state_dict(),
            'best_loss': best_loss,
            'no_improve': no_improve,
        }, CHECKPOINT_FILE)

        # Early stopping
        if no_improve >= patience:
            print(f'\nEarly stopping at epoch {epoch+1} (no improvement for {patience} epochs)')
            break

    print('\nTraining done.')
    print(f'最佳 val_loss: {best_loss:.4f}')
    print(f'模型已保存至: best_model.pth')


if __name__ == '__main__':
    main()
