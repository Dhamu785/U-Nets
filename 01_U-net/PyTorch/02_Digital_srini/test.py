# %% lib imports
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import numpy as np
# %%
path_msk = '/Users/dhamodharan/My-Python/AI-Tutorials/dataset/Types of seg/02_Multi/Y/img_1.png'
img_data = cv2.imread(path_msk, 0)
print(img_data)
# %%
print(np.unique(img_data))
# %%
