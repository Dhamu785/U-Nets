# %% import libs
import torch as t
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from torchvision.utils import draw_segmentation_masks
from PIL import Image
import os
import numpy as np

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

# %% draw mask
dataset_dir_x = '/Users/dhamodharan/My-Python/AI-Tutorials/dataset/X/'
dataset_dir_y = '/Users/dhamodharan/My-Python/AI-Tutorials/dataset/Y/'


img = os.listdir(dataset_dir_x)[0]
img_arr = Image.open(os.path.join(dataset_dir_x, img))
msk = Image.open(os.path.join(dataset_dir_y, img.split('.')[0]+'_mask.gif'))
np_msk = np.array(msk)
np_img = np.array(img_arr)
print(np_msk.shape, np_img.shape)
seg_mask = draw_segmentation_masks(t.from_numpy(np_img).permute(2,0,1), t.from_numpy(np_msk).to(t.bool), colors="blue")
print(seg_mask.shape)
plt.figure(figsize=(15,10))
plt.subplot(1,3,1)
plt.imshow(img_arr)
plt.axis('off')
plt.subplot(1,3,2)
plt.imshow(msk)
plt.axis('off')
plt.subplot(1,3,3)
plt.imshow(seg_mask.permute(1,2,0).numpy())
plt.axis('off')
plt.show()
# %%
