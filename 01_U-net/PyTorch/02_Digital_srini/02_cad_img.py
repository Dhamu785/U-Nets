# %% import libs
import numpy as np
from skimage import color
import matplotlib.pyplot as plt
import cv2
# %%
img_data = cv2.imread(r'C:\Users\dhamu\Documents\Python all\torch_works\01\U-Nets\01_U-net\PyTorch\02_Digital_srini\data\30450.png')
gray_img = img_data[:,:,0]

plt.subplot(1, 2, 1)
plt.imshow(img_data)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(gray_img, cmap='gray')
plt.axis('off')
plt.show()
# %%
