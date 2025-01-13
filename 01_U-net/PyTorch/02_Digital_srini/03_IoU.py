# %% import libs 
# https://youtu.be/BNPW1mYbgS4?si=xjmlOIrjgNDb0xfw
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

# ------------------------------------------------------------------------------------
# %% My IoU
def loss_iou(y_pred, y_true, inf):
    if not inf:
        if not y_pred.requires_grad:
            raise ValueError("y_pred should have gradient tracking")
    
    device = y_pred.device
    # binary_pred = t.where(y_pred <= 0, t.zeros_like(y_pred, device=device, requires_grad=True), t.ones_like(y_pred, device=device, requires_grad=True))
    y_true = t.where(y_true <= 0, t.zeros_like(y_pred, device=device), t.ones_like(y_pred, device=device))
    
    # y_pred = F.sigmoid(y_pred)
    
    intersection = t.abs((y_pred.view((-1)) * y_true.view((-1))).sum().float())
    union = t.abs((y_pred.sum() + y_true.sum()).float())
    # print(f"intersection = {t.abs(intersection)}, union={union}")

    iou = (t.abs(intersection) + 1e-5) / ((union + 1e-5) - t.abs(intersection))
    iou_loss = 1 - iou
    return iou_loss