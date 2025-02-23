# %% lib imports
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import numpy as np

# %% read using pillow
path_msk = '/Users/dhamodharan/My-Python/AI-Tutorials/dataset/Types of seg/02_Multi/Y/img_1.png'
pil_img = Image.open(path_msk)
print(pil_img)
print(np.unique(np.array(pil_img)))
img_data = cv2.imread(path_msk, 0)
print(np.unique(img_data))

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
# %%
