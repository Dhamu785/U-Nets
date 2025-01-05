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
    intersection = np.sum(d1 * d2)
    jaccard = intersection / (np.sum(d1) + np.sum(d2) - intersection)
    return jaccard
# %% print IoU value
print(iou(img1, img2))

# %% calculate the loss
def cal_loss(img1:np.array, img2:np.array) -> float:
    IoU = iou(img1, img2)
    loss = 1 - IoU
    return loss
# %%
loss = cal_loss(img1, img2)
print("Loss = ", loss)