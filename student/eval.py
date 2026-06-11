import sys, os, glob
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import ColorNetMasked
from dataset import ColorDataset


def find_all_targets(root_dir='..'):
    targets = []
    tj = os.path.join(root_dir, 'targets.json')
    if os.path.exists(tj):
        targets.append(tj)
    pattern = os.path.join(root_dir, 'targets_pixiv_*.json')
    targets.extend(sorted(glob.glob(pattern)))
    return targets


def deltaE_ab(lab1, lab2):
    return torch.sqrt(((lab1 - lab2) ** 2).sum(dim=1)).mean().item()


def load_state_dict(path, device):
    """兼容 best_model.pth（dict 格式）和裸 state_dict 两种格式"""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and 'model' in ckpt:
        return ckpt['model']
    return ckpt


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ColorNetMasked().to(device)
    model.load_state_dict(load_state_dict('best_model.pth', device))
    model.eval()

    target_jsons = find_all_targets()
    img_dir = os.path.join('..', 'extracted_imgs', 'imgs')
    val_set = ColorDataset(img_dir, target_jsons, 'val')
    loader = DataLoader(val_set, batch_size=64, shuffle=False)

    if len(val_set) == 0:
        print("验证集为空")
        return

    de_fg, de_bg = 0, 0
    n = 0
    with torch.no_grad():
        for img, lab_fg, lab_bg, _ in loader:
            img, lab_fg, lab_bg = img.to(device), lab_fg.to(device), lab_bg.to(device)
            pred_fg, pred_bg, _ = model(img)
            de_fg += deltaE_ab(pred_fg, lab_fg) * img.size(0)
            de_bg += deltaE_ab(pred_bg, lab_bg) * img.size(0)
            n += img.size(0)

    print(f'验证集样本数: {n}')
    print(f'dE*ab  FG: {de_fg/n:.2f}   BG: {de_bg/n:.2f}')


if __name__ == '__main__':
    main()
