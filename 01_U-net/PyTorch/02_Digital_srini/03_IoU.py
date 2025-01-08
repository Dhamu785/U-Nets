# %% import libs 
import numpy as np
import matplotlib.pyplot as plt
import torch as t

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
    jaccard = intersection / np.sum(d1) + np.sum(d2)
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
# ------------------------------------------------------------------------------------
# %% Generate images with batch size
img_b1 = np.random.randn(16, 64, 64)
img_b2 = np.random.randn(16, 64, 64)
print(img_b1.shape, img_b2.shape)
# %% convert numpy array to torch tensor
torch_b1 = t.from_numpy(img_b1)
torch_b2 = t.from_numpy(img_b2)
print(f"Tensor by torch info : {torch_b1.shape, torch_b2.shape}")
# %% sum only the image from batchs 
print(torch_b1.sum((2,1))) # or print(torch_b1.sum((1,2))) # both r same

# %% 
# ------------------------------------------------------------------------------------
# From kaggle (https://www.kaggle.com/code/iezepov/fast-iou-scoring-metric-in-pytorch-and-numpy)
SMOOTH = 1e-6

def iou_pytorch(outputs: t.Tensor, labels: t.Tensor):
    # You can comment out this line if you are passing tensors of equal shape
    # But if you are passing output from UNet or something it will most probably
    # be with the BATCH x 1 x H x W shape
    outputs = outputs.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W
    
    intersection = (outputs & labels).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    union = (outputs | labels).float().sum((1, 2))         # Will be zzero if both are 0
    
    iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0
    
    thresholded = t.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10  # This is equal to comparing with thresolds
    
    return thresholded  # Or thresholded.mean() if you are interested in average across the batch
