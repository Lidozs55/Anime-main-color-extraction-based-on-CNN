import torch
import torch.nn as nn
import torch.nn.functional as F

class MBConv(nn.Module):
    def __init__(self, in_ch, out_ch, stride, expand_ratio):
        super().__init__()
        hidden_ch = in_ch * expand_ratio
        self.use_res = (stride == 1 and in_ch == out_ch)
        layers = []
        if expand_ratio != 1:
            layers.append(nn.Conv2d(in_ch, hidden_ch, 1, bias=False))
            layers.append(nn.BatchNorm2d(hidden_ch))
            layers.append(nn.ReLU6(inplace=True))
        layers.extend([
            nn.Conv2d(hidden_ch, hidden_ch, 3, stride, 1, groups=hidden_ch, bias=False),
            nn.BatchNorm2d(hidden_ch),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res:
            return x + self.conv(x)
        return self.conv(x)

class ColorNetMasked(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU6(inplace=True)
        )
        self.block1 = MBConv(16, 16, 1, 2)
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
        self.fg_fc = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.2), nn.Linear(32, 3))
        self.bg_fc = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.2), nn.Linear(32, 3))
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
