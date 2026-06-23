"""ColorDataset - 学生模型训练/验证数据集。

输入:
  - img_dir              本地图片目录(单层,不递归)
  - target_jsons         一个或多个 targets JSON 路径
                         (label.py 生成的 targets.json / targets_*.json)
  - split                'train' / 'val'  (前 80% 训练,后 20% 验证,按字典序)
  - img_size             训练 / 评估时的方形边长
  - pixiv_download_dir   Pixiv 图片目录,首次遇到 Pixiv URL 时按需下载
  - use_shadow_removal   是否在 __getitem__ 中应用阴影去除
                         (默认 True,与 student/preview.py 推理时保持一致;
                          否则会出现 train/inference domain shift)
  - cache_dir            阴影去除结果缓存目录(默认: student/.shadow_cache/)
                         阴影去除在 512x512 工作分辨率上进行,结果直接缓存为 PNG,
                         不再 resize 回原图。后续 epoch 直接加载 512x512 PNG,
                         加载速度比原图分辨率快 8-10x。
                         传 None 则禁用缓存。

数据流:
  读图 → [resize→512 → 阴影去除(带缓存)] → RandomHorizontalFlip/Rotation/Crop(训练) → Resize(验证)
        → ToTensor → (3, H, W) float32 in [0, 1]

输出 (__getitem__):
  img_t    (3, H, W) float32 in [0, 1]
  lab_fg   (3,) 前景点 Lab (L, a, b)
  lab_bg   (3,) 背景点 Lab (L, a, b)
"""
import os, json, re, sys, hashlib, tempfile
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
import numpy as np
import cv2

# 引入 graphcolor.shadow(在父目录)
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))
from graphcolor.shadow import ShadowRemover

# 下载 Pixiv 图片用的工具
import urllib.request


IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}


class ColorDataset(Dataset):
    def __init__(self, img_dir, target_jsons, split='train', img_size=128,
                 pixiv_download_dir=None, use_shadow_removal=True,
                 cache_dir="__default__"):
        """
        Args:
            img_dir: 本地图片根目录
            target_jsons: 一个或多个 targets JSON 文件路径（支持 targets.json 和 targets_pixiv_*.json）
            split: 'train' 或 'val'
            img_size: 图片缩放尺寸
            pixiv_download_dir: Pixiv 图片下载目录（默认: 项目根的 pixiv_img/）
            use_shadow_removal: 是否应用阴影去除(必须与 student/preview.py 一致,
                                避免 train/inference 输入分布不一致)
            cache_dir: 阴影去除结果缓存目录。
                       "__default__" → student/.shadow_cache/(默认)
                       None          → 禁用缓存
        """
        self.img_dir = img_dir
        self.img_size = img_size
        self.use_shadow_removal = use_shadow_removal
        # ShadowRemover 实例 — PyTorch DataLoader 多 worker 时,每个 worker 持有自己的实例
        # (通过 worker_init_fn 重建;简单起见,这里 lazy init,__getitem__ 中检查)
        self._shadow_remover = None
        if pixiv_download_dir is None:
            # 项目根的 pixiv_img/ 目录（与 img_dir 同级或在根级）
            # 优先取 img_dir 上一级（即项目根），再向下找 pixiv_img
            parent = os.path.dirname(os.path.abspath(img_dir)) if img_dir else os.getcwd()
            pixiv_download_dir = os.path.join(parent, "pixiv_img")
        self.pixiv_download_dir = pixiv_download_dir
        os.makedirs(self.pixiv_download_dir, exist_ok=True)

        # 阴影去除缓存:首次计算后存 PNG,后续 epoch 直接加载
        if use_shadow_removal and cache_dir == "__default__":
            cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.shadow_cache')
        self.cache_dir = cache_dir if use_shadow_removal else None
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
            # 预计算 ShadowRemover 参数签名,作为缓存键的一部分
            # (参数变更时自动失效旧缓存)
            # fast_mode 是训练默认开启的关键参数,务必计入签名
            _sr = ShadowRemover(enabled=True, fast_mode=True)
            _params = "|".join(str(getattr(_sr, k, ""))
                               for k in ('sigma_ratio', 'shadow_threshold', 'dark_object_ratio',
                                         'use_morphology', 'morph_kernel_size', 'target_l_offset',
                                         'target_l_gain', 'shadow_blend', 'color_threshold',
                                         'ab_compensation_alpha', 'ab_consistency_threshold',
                                         'fast_mode'))
            self._shadow_sig = hashlib.md5(_params.encode()).hexdigest()[:8]
        else:
            self._shadow_sig = None

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
                    else:
                        # 检查本地文件是否在根目录的pixiv_img子文件夹中
                        local_path = os.path.join(self.pixiv_download_dir, key)
                        if os.path.exists(local_path):
                            self.targets[local_path] = val
                        else:
                            print(f"Warning: 未找到图片 {key}")

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

    def _get_shadow_remover(self):
        """Lazy 创建 ShadowRemover;每个 worker 进程第一次调用时构造自己的实例。

        训练时默认使用 fast_mode=True (跳过连通块+ab 检查),
        与 student/preview.py 保持一致,避免 train/inference domain shift。
        """
        if self._shadow_remover is None:
            self._shadow_remover = ShadowRemover(enabled=True, fast_mode=True)
        return self._shadow_remover

    def _shadow_cache_path(self, path):
        """生成阴影去除缓存的文件路径。

        缓存键 = 原图路径 + mtime + size + 阴影参数签名。
        原图修改或阴影参数变更时自动失效。
        """
        if not self.cache_dir:
            return None
        try:
            stat = os.stat(path)
        except OSError:
            return None
        key_str = f"{path}|{stat.st_mtime}|{stat.st_size}|{self._shadow_sig}"
        cache_key = hashlib.md5(key_str.encode()).hexdigest()[:16]
        return os.path.join(self.cache_dir, f"{cache_key}.png")

    @staticmethod
    def _atomic_save_png(img, cache_path):
        """原子写入 PNG(先写临时文件再 rename,避免多 worker 并发损坏)。"""
        try:
            tmp_fd, tmp_path = tempfile.mkstemp(
                suffix='.png', dir=os.path.dirname(cache_path))
            os.close(tmp_fd)
            img.save(tmp_path, format='PNG')
            os.replace(tmp_path, cache_path)
        except (OSError, IOError):
            pass  # 缓存写入失败不影响训练

    def __getitem__(self, idx):
        path = self.paths[idx]

        if self.use_shadow_removal:
            cache_path = self._shadow_cache_path(path)
            if cache_path and os.path.exists(cache_path):
                # 缓存命中:直接加载已去阴影的图片(512x512 或原图≤512)
                img = Image.open(cache_path).convert('RGB')
            else:
                # 缓存未命中:读原图 → resize 到 512 → 去阴影 → 直接缓存
                # 不再 resize 回原图:512x512 上 RandomResizedCrop 仍有 4x 缩放范围,
                # 且缓存文件小 3.8x、加载快 8-10x
                img = Image.open(path).convert('RGB')
                bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                if max(bgr.shape[:2]) > 512:
                    bgr = cv2.resize(bgr, (512, 512), interpolation=cv2.INTER_AREA)
                bgr = self._get_shadow_remover().remove(bgr)
                img = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                if cache_path:
                    self._atomic_save_png(img, cache_path)
        else:
            img = Image.open(path).convert('RGB')

        img_t = self.transform(img)
        tgt = self.targets[path]
        return img_t, torch.tensor([tgt['L_fg'], tgt['a_fg'], tgt['b_fg']]), torch.tensor([tgt['L_bg'], tgt['a_bg'], tgt['b_bg']])
