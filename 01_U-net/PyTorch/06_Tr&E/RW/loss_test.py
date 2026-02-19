# %% imports
import torch as t
import torch.nn.functional as F
import cv2
import matplotlib.pyplot as plt
# %%
img_pth = "C:\\Users\\dhamu\\Downloads\\Interns unet\\Loss testing\\4349.png"
img_data = cv2.imread(img_pth, 0)
device = 'cuda'
tensor = t.tensor(img_data).unsqueeze(-1).permute(2, 0, 1).to(dtype=t.float32, device=device)
print(tensor.shape)
print(img_data.shape)
# %%
sobel_x = t.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], device=device, dtype=t.float32).view(1,1,3,3)
sobel_y = t.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], device=device, dtype=t.float32).view(1,1,3,3)
grad_x = F.conv2d(tensor, sobel_x, padding=1)
grad_y = F.conv2d(tensor, sobel_y, padding=1)

# grad_x = grad_x / grad_x.max().item()
# grad_y = grad_y / grad_y.max().item()

# %%
plt.imshow(grad_x.permute(1,2,0).to('cpu'), cmap='binary')
plt.show()
plt.imshow(grad_y.permute(1,2,0).to('cpu'), cmap='binary')
plt.show()
# %%
grad_x.unique()
# %%
edge = t.sqrt(grad_x**2 + grad_y**2 + 1e-6)
print(edge)
# %%
edge.shape
# %%
edge.sum()
# %%
F.l1_loss(edge, edge+1)
# %%
