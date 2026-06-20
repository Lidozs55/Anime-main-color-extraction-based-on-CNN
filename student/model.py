"""
学生模型定义 —— ColorNet-Masked。

模型结构(共约 3MB):
    输入:  RGB 图像, (B, 3, 128, 128)
       ↓
    stem:  3x3 conv, stride 2                 → (B, 16, 64, 64)
       ↓
    block1: MBConv(16, 16, 1, expand=1)
    block2: MBConv(16, 24, 2, expand=4)      ↓ /2
    block3: MBConv(24, 24, 1, expand=4)
    block4: MBConv(24, 32, 2, expand=6)      ↓ /2
    block5: MBConv(32, 32, 1, expand=6)
       ↓
    fusion: 1x1 conv → 64ch + ReLU6           → (B, 64, 16, 16)
       ↓
    mask_head: 1x1 conv + Sigmoid             → (B, 1, 16, 16)  软主体掩膜
       ↓
    软池化: 分别用 mask / (1-mask) 加权求和    → (B, 64)
       ↓
    fg_fc:  64 → 32 → 3                       → (B, 3)  前景 Lab
    bg_fc:  64 → 32 → 3                       → (B, 3)  背景 Lab

输出 (forward):
    (out_fg, out_bg, mask)
        out_fg / out_bg:  (B, 3) Lab, L∈[0,100] (sigmoid), a/b∈[-128,127] (tanh)
        mask:             (B, 1, 16, 16) soft 主体掩膜(供可视化/正则)

设计上模仿 MobileNetV2 / EfficientNet 的 MBConv,在 CPU 上也能
单张 ~30ms 完成推理;mask 分支显式拆开 fg / bg,提供可解释性。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MBConv(nn.Module):
    """MobileNetV2 风格的倒残差块(MBConv)。

    通道数变化:  in → in·expand (depthwise) → out
    激活函数:    ReLU6(对低精度友好)
    残差:        stride==1 且 in==out 时启用 add 短路
    """
    def __init__(self, in_ch, out_ch, stride, expand_ratio):
        super().__init__()
        hidden_ch = in_ch * expand_ratio
        # 残差仅在空间尺寸和通道数都不变时成立
        self.use_res = (stride == 1 and in_ch == out_ch)
        layers = []
        if expand_ratio != 1:
            # 1x1 升维
            layers.append(nn.Conv2d(in_ch, hidden_ch, 1, bias=False))
            layers.append(nn.BatchNorm2d(hidden_ch))
            layers.append(nn.ReLU6(inplace=True))
        # 3x3 depthwise 卷积
        layers.extend([
            nn.Conv2d(hidden_ch, hidden_ch, 3, stride, 1, groups=hidden_ch, bias=False),
            nn.BatchNorm2d(hidden_ch),
            nn.ReLU6(inplace=True),
            # 1x1 降维(linear, 不带激活)
            nn.Conv2d(hidden_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res:
            return x + self.conv(x)
        return self.conv(x)


class ColorNetMasked(nn.Module):
    """ColorNet-Masked: 带软掩膜的轻量主色预测 CNN。

    详见模块级 docstring。
    """
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU6(inplace=True)
        )
        self.block1 = MBConv(16, 16, 1, 1)
        self.block2 = MBConv(16, 24, 2, 4)
        self.block3 = MBConv(24, 24, 1, 4)
        self.block4 = MBConv(24, 32, 2, 6)
        self.block5 = MBConv(32, 32, 1, 6)
        self.fusion = nn.Sequential(
            nn.Conv2d(32, 64, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU6(inplace=True)
        )
        self.mask_head = nn.Sequential(
            nn.Conv2d(64, 1, 1),
            nn.Sigmoid()
        )
        self.fg_fc = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 3))
        self.bg_fc = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 3))
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.fusion(x)               # (B,64,H/8,W/8)
        mask = self.mask_head(x)         # (B,1,H/8,W/8)
        eps = 1e-6
        fg_feat = (mask * x).sum(dim=[2,3]) / (mask.sum(dim=[2,3]) + eps)
        bg_feat = ((1-mask) * x).sum(dim=[2,3]) / ((1-mask).sum(dim=[2,3]) + eps)
        out_fg = self.fg_fc(fg_feat)
        out_bg = self.bg_fc(bg_feat)
        L_fg = 100.0 * torch.sigmoid(out_fg[:, 0:1])
        ab_fg = 128.0 * torch.tanh(out_fg[:, 1:3])
        L_bg = 100.0 * torch.sigmoid(out_bg[:, 0:1])
        ab_bg = 128.0 * torch.tanh(out_bg[:, 1:3])
        return torch.cat([L_fg, ab_fg], dim=1), torch.cat([L_bg, ab_bg], dim=1), mask
