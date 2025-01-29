# %% import libs
import torch as t
import torchvision
from torch.utils.data import DataLoader
from dataset import carvana

# %%
def save_checkpoint(statr, filename="checkpoint.tar"):
    ...