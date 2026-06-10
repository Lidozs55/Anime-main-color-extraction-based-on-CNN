import sys, os
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import ColorNetMasked

model = ColorNetMasked()
model.load_state_dict(torch.load('best_model.pth', map_location='cpu'))
model.eval()
dummy = torch.randn(1, 3, 128, 128)
traced = torch.jit.trace(model, dummy)
traced.save('colornet_masked.pt')
print('Exported to colornet_masked.pt')
