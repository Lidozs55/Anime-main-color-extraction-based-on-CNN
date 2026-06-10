根据已有项目结构，你需完成蒸馏数据生成、轻量CNN训练与导出。请严格按以下步骤执行，所有交互操作均在控制台完成。

---

### 1. 生成蒸馏软目标（targets.json）

**输入：** `outputs/results.json`（假设已由 graphcolor 生成）  
**输出：** `targets.json`，包含每张图片经人工校验后的前景/背景主色 Lab 值（各一个）。

#### 1.1 读取与解析
- 读取 `outputs/results.json`
- 若文件不存在，请先运行 `python main.py` 生成结果，完成后继续。

#### 1.2 对每张图片执行交互修正（前景与背景独立处理）
对前景、背景**分别**执行以下步骤：
1. 将颜色按 `score` 降序排序，得到列表 `[C1, C2, C3]` 及其评分。
2. 计算第一名与第二名的评分差距（相对差）：`gap = (score1 - score2) / score1`。  
   - 若 `gap >= 0.05`：不做人工干预，保持当前顺序。  
   - 若 `gap < 0.05`：暂停并请人类重新排序。  
     * 生成预览图：将原始图与三个主色色块合成形成一张预览图，输出至 `outputs/previews/` 文件夹。
     * 进行提示：图片文件名、当前排序的三个颜色 Lab 值（含编号 1,2,3），并提示“请查看 `outputs/previews/` 中对应图片的预览图，然后输入你认定的色彩排序（用空格分隔三个数字，如 `2 1 3` 表示将第二个颜色作为第一名，第一个作为第二名，第三个不变）：”  
     * 读取用户输入的三个整数（1~3 的排列）。  
     * 根据输入重新映射评分：新排序的第一名继承原 `score1`，第二名继承 `score2`，第三名继承 `score3`。**颜色顺序按用户指定调整，评分随位次移动**。
3. 完成可能的调整后，**对排名第一的评分乘以系数 2**，其余评分不变。
4. 对调整后的三个评分应用 softmax（温度 T=2.0）：  
   `w_i = exp(score_i / 2) / Σ exp(score_j / 2)`  
5. 计算软目标 Lab：  
   `L = Σ w_i * L_i`,  `a = Σ w_i * a_i`,  `b = Σ w_i * b_i`

#### 1.3 保存 targets.json
将每张图片的前景、背景加权 Lab 保存为字典，键为图片文件名（不含路径），值为：
```json
{
  "L_fg": 52.3, "a_fg": 20.1, "b_fg": -31.2,
  "L_bg": 78.5, "a_bg": -5.0, "b_bg": 12.7
}
```
写入项目根目录下的 `targets.json`。

---

### 2. 学生模型训练

#### 2.1 文件组织
在项目根目录创建 `student/` 文件夹，放入以下脚本：
- `model.py`：ColorNet-Masked 模型定义  
- `dataset.py`：数据集类  
- `train.py`：训练主程序  
- `eval.py`：评估脚本  
- `export.py`：导出脚本  

#### 2.2 模型架构 (model.py)
```python
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
```

#### 2.3 数据集 (dataset.py)
```python
import os, json
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

class ColorDataset(Dataset):
    def __init__(self, img_dir, target_json, split='train', img_size=128):
        self.img_dir = img_dir
        self.img_size = img_size
        with open(target_json) as f:
            self.targets = json.load(f)
        self.names = sorted(self.targets.keys())
        n = len(self.names)
        split_idx = int(n * 0.8)
        if split == 'train':
            self.names = self.names[:split_idx]
        else:
            self.names = self.names[split_idx:]
        if split == 'train':
            self.transform = T.Compose([
                T.RandomHorizontalFlip(0.5),
                T.RandomRotation(15, fill=0),
                T.RandomResizedCrop(img_size, scale=(0.8,1.0), ratio=(0.9,1.1)),
                T.ToTensor(),
            ])
        else:
            self.transform = T.Compose([
                T.Resize((img_size, img_size)),
                T.ToTensor(),
            ])

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        img = Image.open(os.path.join(self.img_dir, name)).convert('RGB')
        img_t = self.transform(img)
        tgt = self.targets[name]
        return img_t, torch.tensor([tgt['L_fg'], tgt['a_fg'], tgt['b_fg']]), torch.tensor([tgt['L_bg'], tgt['a_bg'], tgt['b_bg']])
```

#### 2.4 训练 (train.py)
```python
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from model import ColorNetMasked
from dataset import ColorDataset

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ColorNetMasked().to(device)
    train_set = ColorDataset('../extracted_imgs', '../targets.json', 'train')
    val_set   = ColorDataset('../extracted_imgs', '../targets.json', 'val')
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_set, batch_size=64, shuffle=False, num_workers=4)

    opt = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=60)
    mse = nn.MSELoss()
    best_loss = float('inf')
    for epoch in range(60):
        model.train()
        for img, lab_fg, lab_bg in train_loader:
            img, lab_fg, lab_bg = img.to(device), lab_fg.to(device), lab_bg.to(device)
            pred_fg, pred_bg, mask = model(img)
            loss_col = mse(pred_fg, lab_fg) + mse(pred_bg, lab_bg)
            mask_mean = mask.mean(dim=[1,2,3])
            loss_mask = ((mask_mean - 0.4)**2).mean()
            loss = loss_col + 0.1 * loss_mask
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for img, lab_fg, lab_bg in val_loader:
                img, lab_fg, lab_bg = img.to(device), lab_fg.to(device), lab_bg.to(device)
                pred_fg, pred_bg, _ = model(img)
                val_loss += (mse(pred_fg, lab_fg) + mse(pred_bg, lab_bg)).item() * img.size(0)
        val_loss /= len(val_set)
        print(f'Epoch {epoch+1:02d}  val_loss={val_loss:.4f}')
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
    print('Training done.')

if __name__ == '__main__':
    main()
```

#### 2.5 评估 (eval.py)
```python
import torch
from model import ColorNetMasked
from dataset import ColorDataset
from torch.utils.data import DataLoader

def deltaE_ab(lab1, lab2):
    return torch.sqrt(((lab1 - lab2)**2).sum(dim=1)).mean().item()

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ColorNetMasked().to(device)
    model.load_state_dict(torch.load('best_model.pth', map_location=device))
    model.eval()
    val_set = ColorDataset('../extracted_imgs', '../targets.json', 'val')
    loader = DataLoader(val_set, batch_size=64, shuffle=False)
    de_fg, de_bg = 0, 0
    n = 0
    with torch.no_grad():
        for img, lab_fg, lab_bg in loader:
            img, lab_fg, lab_bg = img.to(device), lab_fg.to(device), lab_bg.to(device)
            pred_fg, pred_bg, _ = model(img)
            de_fg += deltaE_ab(pred_fg, lab_fg) * img.size(0)
            de_bg += deltaE_ab(pred_bg, lab_bg) * img.size(0)
            n += img.size(0)
    print(f'ΔE*ab  FG: {de_fg/n:.2f}   BG: {de_bg/n:.2f}')

if __name__ == '__main__':
    main()
```

#### 2.6 导出 (export.py)
```python
import torch
from model import ColorNetMasked

model = ColorNetMasked()
model.load_state_dict(torch.load('best_model.pth', map_location='cpu'))
model.eval()
dummy = torch.randn(1, 3, 128, 128)
traced = torch.jit.trace(model, dummy)
traced.save('colornet_masked.pt')
print('Exported to colornet_masked.pt')
```

---

### 3. 执行顺序
1. 确保 `extracted_imgs/` 包含所有训练图片，`outputs/results.json` 存在。
2. 运行**目标生成脚本**（自行编写 `generate_targets.py` 实现第 1 节逻辑），生成 `targets.json`。
3. 进入 `student/` 目录，依次执行：
   - `python train.py`
   - `python eval.py`
   - `python export.py`
4. 最终模型文件为 `student/colornet_masked.pt`，可直接部署到边缘设备。

---

**注意事项：**
- 所有图片输入统一缩放至 128×128，不做色彩增强。
- 若训练中掩膜均值持续接近 0 或 1，可适当增大 `loss_mask` 系数（原 0.1）。
- 人工交互环节需查看 `outputs/previews/` 中的预览图，确保控制台能正常显示中文提示。