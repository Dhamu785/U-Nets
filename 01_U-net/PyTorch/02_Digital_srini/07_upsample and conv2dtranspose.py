# %% import libs
from torch import nn
import torch as t
# %%
DEVICE = 'cuda' if t.cuda.is_available() else 'cpu'
print(f'Available device = {DEVICE}')

kernel = t.tensor([[[[1,2,3], [4,5,6], [1,2,3]]]], dtype=t.float32, device=DEVICE)
print(kernel)
print(kernel.shape)
# %%
img = t.ones((1,1,4,4), dtype=t.float32, device=DEVICE)
print(img)
# %%
trans = nn.functional.conv_transpose2d(img, kernel, stride=1, padding=1)
trans.shape
