# %% import libs
import torch as t
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

# %% make random tensors
DEVICE = 'cuda' if t.cuda.is_available() else 'cpu'
print(f"Available device = {DEVICE}")
imgs = t.randn((10, 3, 64, 64), device=DEVICE)
print(f"Shape of imgs = {imgs.shape}")

# %%
# grid = make_grid(imgs, 2)
grid = make_grid(imgs, 5, 2, pad_value=2).moveaxis(0, 2)
print("Shape of grid = ", grid.shape)
f, axs = plt.subplots(figsize=(5,10))
axs.set_axis_off()
# plt.close()
axs.imshow(grid)