# %% import libs
import torch as t
import torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

import random

from PIL import Image

# %% GPU checks
DEVICE = 'cuda' if t.cuda.is_available() else 'cpu'
print(f"CUDA version = {t.version.cuda}")
print(f"CuDNN version = {t.backends.cudnn.version()}")
print(f"CuDNN enabled = {t.backends.cudnn.enabled}")

# %% Load model
# model_path = '/Users/dhamodharan/My-Python/AI-Tutorials/Models/U-Net_resnet101.pt'
model_path = "C:\\Users\\dhamu\\Documents\\Python all\\torch_works\\01\\Models\\Carvana_Image_wt.pt"
st_dict = t.load(model_path, map_location=t.device(DEVICE), weights_only=True)

layers = []
for i in st_dict:
    if 'weight' in i:
        layers.append((i, st_dict[i].shape))
print(f"Total layers = {len(layers)}")

# %% 
transform_img = transforms.Compose([
            transforms.Resize((512,512)),
            transforms.ToTensor()
        ])
img_path = r"C:\Users\dhamu\Documents\Python all\torch_works\01\dataset\01_semantic segmentation\X\0cdf5b5d0ce1_03.jpg"
pil_img = Image.open(img_path).convert('RGB')

transformed = transform_img(pil_img).unsqueeze(0)
print(transformed.shape)
# %%
def conv(inputs, layer_no):
    res = F.conv2d(inputs.to(DEVICE), st_dict[layers[layer_no][0]], padding=1).squeeze(0)
    print(res.shape)
    return res

# %%
randoms = random.sample(range(0,64), 5)
# %%
def plot_(result, layer):
    plt.figure(figsize=(2,5))
    for i in range(len(randoms)):
        plt.subplot(5,1,i+1)
        plt.imshow(result[:,:,randoms[i]], cmap='binary')
        plt.axis('off')
    plt.suptitle(f'Layer-{layer}', size='medium', y=0.92)
    plt.savefig(f'layer-{layer}.png')
    # plt.show()

# %%
res = transform_img(pil_img).unsqueeze(0)

for i,j in enumerate(layers):
    res = conv(res, i)
    ress = res.permute(1,2,0).to('cpu')
    print(ress.shape)
    plot_(ress, i)
# %%
