import sys, os
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import ColorNetMasked

model = ColorNetMasked()
ckpt = torch.load('best_model.pth', map_location='cpu', weights_only=False)
model.load_state_dict(ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt)
model.eval()
dummy = torch.randn(1, 3, 128, 128)
traced = torch.jit.trace(model, dummy)
traced.save('colornet_masked.pt')
print('Exported to colornet_masked.pt')
