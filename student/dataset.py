import os, json, re
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

# 下载 Pixiv 图片用的工具
import urllib.request


IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}


class ColorDataset(Dataset):
    def __init__(self, img_dir, target_jsons, split='train', img_size=128, pixiv_download_dir=None):
        """
        Args:
            img_dir: 本地图片根目录
            target_jsons: 一个或多个 targets JSON 文件路径（支持 targets.json 和 targets_pixiv_*.json）
            split: 'train' 或 'val'
            img_size: 图片缩放尺寸
            pixiv_download_dir: Pixiv 图片下载目录（默认: img_dir/pixiv_imgs）
        """
        self.img_dir = img_dir
        self.img_size = img_size
        self.pixiv_download_dir = pixiv_download_dir or os.path.join(img_dir, "pixiv_imgs")
        os.makedirs(self.pixiv_download_dir, exist_ok=True)

        # 合并所有 JSON 的 targets
        self.targets = {}
        self.pixiv_urls = {}  # URL -> 本地路径映射

        if isinstance(target_jsons, str):
            target_jsons = [target_jsons]

        for tj in target_jsons:
            with open(tj, encoding='utf-8') as f:
                data = json.load(f)
            for key, val in data.items():
                # 判断是文件名（targets.json）还是 URL（targets_pixiv_*.json）
                if key.startswith('http'):
                    # Pixiv URL 格式，先找本地再下载
                    local_path = self._resolve_pixiv_image(key)
                    if local_path and os.path.exists(local_path):
                        self.targets[local_path] = val
                        self.pixiv_urls[key] = local_path
                else:
                    # 文件名格式
                    local_path = os.path.join(img_dir, key)
                    if os.path.exists(local_path):
                        self.targets[local_path] = val

        self.paths = sorted(self.targets.keys())
        n = len(self.paths)
        split_idx = max(1, int(n * 0.8))
        if split == 'train':
            self.paths = self.paths[:split_idx]
        else:
            self.paths = self.paths[split_idx:]

        if split == 'train':
            self.transform = T.Compose([
                T.RandomHorizontalFlip(0.5),
                T.RandomRotation(15, fill=0),
                T.RandomResizedCrop(img_size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
                T.ToTensor(),
            ])
        else:
            self.transform = T.Compose([
                T.Resize((img_size, img_size)),
                T.ToTensor(),
            ])

    def _url_to_filename(self, url):
        """从 URL 提取可能的文件名"""
        basename = url.split('/')[-1]
        # e.g. 123456_p0.jpg
        if any(basename.lower().endswith(ext) for ext in IMAGE_EXTS):
            return basename
        # e.g. https://www.pixiv.net/en/artworks/123456 - extract artwork ID
        match = re.search(r'artworks/(\d+)', url)
        if match:
            return f"{match.group(1)}.jpg"  # fallback
        return None

    def _resolve_pixiv_image(self, url):
        """下载 Pixiv 图片到本地，先找本地已存在的文件"""
        # 1. 尝试从 URL 提取文件名，在本地目录中查找
        filename = self._url_to_filename(url)
        if filename:
            local_path = os.path.join(self.pixiv_download_dir, filename)
            if os.path.exists(local_path):
                return local_path

        # 2. 在本地目录中搜索匹配的文件（按 pid 前缀）
        if filename:
            pid_prefix = filename.split('_')[0]  # e.g. "123456"
            pattern = os.path.join(self.pixiv_download_dir, f"{pid_prefix}*")
            matches = []
            for ext in IMAGE_EXTS:
                matches.extend([os.path.join(self.pixiv_download_dir, f"{pid_prefix}*{ext}"),
                                os.path.join(self.pixiv_download_dir, f"{pid_prefix}*{ext.upper()}")])
            import glob
            for pat in matches:
                found = glob.glob(pat)
                if found:
                    return found[0]

        # 3. 本地没有，下载
        if filename:
            local_path = os.path.join(self.pixiv_download_dir, filename)
        else:
            local_path = os.path.join(self.pixiv_download_dir, f"pixiv_unknown_{hash(url) % 1000000}.jpg")

        if os.path.exists(local_path):
            return local_path

        try:
            req = urllib.request.Request(url, headers={
                "User-Agent": "Mozilla/5.0",
                "Referer": "https://www.pixiv.net/",
            })
            with urllib.request.urlopen(req, timeout=30) as resp:
                img_data = resp.read()
            with open(local_path, "wb") as f:
                f.write(img_data)
            # 验证图片
            img = Image.open(local_path)
            img.verify()
            img.close()
            print(f"  下载 Pixiv 图片: {os.path.basename(local_path)}")
            return local_path
        except Exception as e:
            print(f"  下载失败 {os.path.basename(local_path)}: {e}")
            return None

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = Image.open(path).convert('RGB')
        img_t = self.transform(img)
        tgt = self.targets[path]
        return img_t, torch.tensor([tgt['L_fg'], tgt['a_fg'], tgt['b_fg']]), torch.tensor([tgt['L_bg'], tgt['a_bg'], tgt['b_bg']])
