"""
学生模型评估脚本:在验证集上报告 dE*ab 指标。

用法:
    cd student
    python eval.py

dE*ab 越小表示预测 Lab 与教师标签 Lab 在感知空间上越接近,
常用于主色/调色类模型的精度评估。一般认为:
    dE*ab < 1   肉眼几乎不可分辨
    dE*ab < 3   高质量
    dE*ab < 5   可接受
"""
import sys, os, glob
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import ColorNetMasked
from dataset import ColorDataset


# 项目根目录
PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
DEFAULT_IMG_DIR = os.path.join(PROJECT_ROOT, 'img')
DEFAULT_PIXIV_DIR = os.path.join(PROJECT_ROOT, 'pixiv_img')


def find_all_targets(root_dir=None):
    """同 train.py:扫描项目根目录所有 targets*.json 文件。"""
    targets = []
    if root_dir is None:
        root_dir = PROJECT_ROOT
    tj = os.path.join(root_dir, 'targets.json')
    if os.path.exists(tj):
        targets.append(tj)
    for pat in ('targets_pixiv_*.json', 'targets_*.json'):
        for f in sorted(glob.glob(os.path.join(root_dir, pat))):
            if f not in targets:
                targets.append(f)
    return targets


def deltaE_ab(lab1, lab2):
    """简化的 dE*ab(欧氏距离版本,未做白点归一化)。

    Args:
        lab1, lab2: (B, 3) Lab 张量
    Returns:
        整批的平均 dE*ab(scalar)
    """
    return torch.sqrt(((lab1 - lab2) ** 2).sum(dim=1)).mean().item()


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ColorNetMasked().to(device)
    model.load_state_dict(torch.load('best_model.pth', map_location=device))
    model.eval()

    target_jsons = find_all_targets()
    img_dir = DEFAULT_IMG_DIR
    val_set = ColorDataset(img_dir, target_jsons, 'val', pixiv_download_dir=DEFAULT_PIXIV_DIR)
    loader = DataLoader(val_set, batch_size=64, shuffle=False)

    if len(val_set) == 0:
        print("验证集为空")
        return

    de_fg, de_bg = 0, 0
    n = 0
    with torch.no_grad():
        for img, lab_fg, lab_bg in loader:
            img, lab_fg, lab_bg = img.to(device), lab_fg.to(device), lab_bg.to(device)
            pred_fg, pred_bg, _ = model(img)
            de_fg += deltaE_ab(pred_fg, lab_fg) * img.size(0)
            de_bg += deltaE_ab(pred_bg, lab_bg) * img.size(0)
            n += img.size(0)

    print(f'验证集样本数: {n}')
    print(f'dE*ab  FG: {de_fg/n:.2f}   BG: {de_bg/n:.2f}')


if __name__ == '__main__':
    main()
