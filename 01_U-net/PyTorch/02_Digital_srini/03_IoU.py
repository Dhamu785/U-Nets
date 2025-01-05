# %% import libs 
import numpy as np
import os
import matplotlib.pyplot as plt

#%% create img''s
img1 = np.random.randn(64,64)
img2 = np.random.randn(64, 64)

img1[img1 > 0.5] = 1
img1[img1 <= 0.5] = 0

img2[img2 > 0.5] = 1
img2[img2 <= 0.5] = 0
print(img1.shape, img2.shape)
print(np.unique(img1), np.unique(img2))
#%% plot the created random image
plt.imshow(img1, cmap='gray')
plt.axis('off')
plt.title('Img1')
plt.show()
plt.imshow(img2, cmap='gray')
plt.title('Img2')
plt.axis('off')
plt.show()
# %% Jaccard loss function(IoU)
def iou(img1:np.array, img2:np.array) -> float:
    d1 = np.ravel(img1)
    d2 = np.ravel(img2)
    return d1.shape, d2.shape
# %%
print(iou(img1, img2))
# %%
