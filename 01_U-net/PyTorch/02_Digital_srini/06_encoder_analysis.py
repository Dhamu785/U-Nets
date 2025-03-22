# %% import libs
import torch as t
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

import os

import segmentation_models_pytorch as smp
# %% GPU checks
DEVICE = 'cuda' if t.cuda.is_available() else 'cpu'
print(f"CUDA version = {t.version.cuda}")
print(f"CuDNN version = {t.backends.cudnn.version()}")
print(f"CuDNN enabled = {t.backends.cudnn.enabled}")

# %% Load model
model_path = '/Users/dhamodharan/My-Python/AI-Tutorials/Models/U-Net_resnet101.pt'
# model_path = "C:\\Users\\dhamu\\Documents\\Python all\\torch_works\\01\\Models\\U-net_efficientnet.pt"
st_dict = t.load(model_path, map_location=t.device(DEVICE))

ENCODER_NAME = 'resnet101'
ENCODER_WEIGHT = 'imagenet'
NUM_OF_CLS = 1

model = smp.Unet(encoder_name=ENCODER_NAME, encoder_weights=ENCODER_WEIGHT, in_channels=3, classes=NUM_OF_CLS).to(DEVICE)
model = model.load_state_dict(state_dict=st_dict)

# %% 
res = F.conv2d(t.randn(1,3,512,512, device=DEVICE), st_dict['encoder._conv_stem.weight'])
# %%
res.shape
# %%
layers = []
for i in st_dict:
    if 'weight' in i:
        layers.append(i)

print(f"Total layers = {len(layers)}")
# %%
