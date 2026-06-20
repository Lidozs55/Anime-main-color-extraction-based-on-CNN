"""
学生模型导出为 TorchScript。

用法:
    cd student
    python export.py
    # 产出 colornet_masked.pt (单文件,可被 C++/LibTorch/Python 一致加载)

TorchScript 相比纯 state_dict 的好处:
  - 不再需要源码中 class 的依赖(便于部署到没有 model.py 的环境)
  - 支持 torch.jit.load() 直接反序列化
  - 与 ONNX 不同,无需额外运行时;同时保留了 dynamic shape 能力
"""
import sys, os
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import ColorNetMasked

# 加载已训练好的最优权重
model = ColorNetMasked()
model.load_state_dict(torch.load('best_model.pth', map_location='cpu'))
model.eval()

# 用一张随机 dummy 走一遍前向,完成 tracing
dummy = torch.randn(1, 3, 128, 128)
traced = torch.jit.trace(model, dummy)
traced.save('colornet_masked.pt')
print('Exported to colornet_masked.pt')
