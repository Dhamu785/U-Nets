# %% lib imports
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import numpy as np

# %% read using pillow and cv2
path_msk = '/Users/dhamodharan/My-Python/AI-Tutorials/dataset/Types of seg/02_Multi/Y/img_1.png'
pil_img = Image.open(path_msk)
print(pil_img)
print(np.unique(np.array(pil_img)))
img_data = cv2.imread(path_msk, 0)
print(np.unique(img_data))
# =======================================================================================
# %% Label encoder + onehot encoder test
import torch as t
from sklearn.preprocessing import LabelEncoder
import numpy as np
from torchvision import transforms

sample = np.array([['cat', 'dog', 'fish'],['cat', 'fish', 'fish']]).reshape(-1)
print(f"Samples shape = {sample}")
labels = LabelEncoder()
lbl = labels.fit_transform(sample)
print(lbl)
# %%
onehot = t.nn.functional.one_hot(t.tensor(lbl), 3)
print(onehot)
# %%
print(onehot.reshape(2,3, -1))
print(onehot.reshape(2,3, -1).shape)
# %%
print(onehot.reshape(-1,2,3))
transposed = np.transpose(onehot.reshape(2,3,-1), (2,0,1))
print(transposed)
transposed[:, 1, 2]
# %%
np.transpose(onehot.reshape(2,3,-1), (2,0,1)).shape
# =======================================================================================

# %% Calculate the class weights
from sklearn.utils import class_weight
import numpy as np
# %%
s1 = np.array([1,1,1,1,1,1,2,2,2,3])
print(f"Sample1 = {s1}")
cls_wgt = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(s1), y=s1)
print(cls_wgt)
# =======================================================================================

# %% Calculate argmax for b, c, h, w
import torch as t
# %%
s1 = t.randn((1,4,4,4))
print("s1 shape = ", s1.shape)
print("Datapoints = ", s1)
argmax_res = t.argmax(s1, 1)
print(f"Shape of the argmax = {argmax_res.shape}")
print(f"Result of argmax = {argmax_res}")
# =======================================================================================

# %%
from utils import get_loaders
from train import main
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

# %%
trainimg = "C:\\Users\\dhamu\\Documents\\Python all\\torch_works\\01\\dataset\\02_instance segmentation\\images"
trainmsk = "C:\\Users\\dhamu\\Documents\\Python all\\torch_works\\01\\dataset\\02_instance segmentation\\masks"
testnimg = "C:\\Users\\dhamu\\Documents\\Python all\\torch_works\\01\\dataset\\02_instance segmentation\\images"
testnmsk = "C:\\Users\\dhamu\\Documents\\Python all\\torch_works\\01\\dataset\\02_instance segmentation\\masks"
train_transform = A.Compose([
        A.Resize(height = 256, width = 256),
        A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
        ToTensorV2()
    ])

val_transform = A.Compose([
    A.Resize(height=256, width=256),
    A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
    ToTensorV2()
])
train, test = get_loaders(trainimg, trainmsk, trainimg, trainmsk, 8, train_transform, val_transform, 0, False)

# %%
i,j = next(iter(train))
print(f"unique of y = {np.unique(j)}")
print(f"X: Min = {i.min()}, Max = {i.max()}")
print(f"{i.shape = }, {j.shape = }")
