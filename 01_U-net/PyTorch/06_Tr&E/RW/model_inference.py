# %%
import torch as t
from torchvision import transforms
from PIL import Image
from loss import Edge_IoU
# %%
img_x = "C:\\Users\\dhamu\\Downloads\\Interns unet\\Loss testing\\test\\X.png"
img_y = "C:\\Users\\dhamu\\Downloads\\Interns unet\\Loss testing\\test\\Y.png"

x = Image.open(img_x).convert('L')
y = Image.open(img_y).convert('L')
to_tensor = transforms.ToTensor()
x1 = to_tensor(x).unsqueeze(0).to(device='cuda')
y1 = to_tensor(y).unsqueeze(0).requires_grad_(True).to(device='cuda')
# %%
x1.shape
# %%
loss = Edge_IoU(0.3, 0.7, 'cuda')

# %%
loss(x1, y1)
# %%
y1.requires_grad
# %%
