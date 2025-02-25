import os
import torch as t

import cv2
import numpy as np
from sklearn.utils import class_weight
import train

DEVICE = 'cuda' if t.cuda.is_available() else 'cpu'
print(f"Available device = {DEVICE}")
print(f"Version of cuda = {t.version.cuda}")
print(f"CuDNN vrsion = {t.backends.cudnn.version()}")
print(f"is enabled = {t.backends.cudnn.enabled}")
mask_path = 'C:\\Users\\dhamu\\Documents\\Python all\\torch_works\\01\\dataset\\02_instance segmentation\\masks'
image_path = 'C:\\Users\\dhamu\\Documents\\Python all\\torch_works\\01\\dataset\\02_instance segmentation\\images'
print(f"Total files in mask_path = {len(os.listdir(mask_path))}\nTotal files in image_path = {len(os.listdir(image_path))}")

counts = np.array([0, 0, 0, 0])
masks = os.listdir(mask_path)
img_array = []
for i in masks:
    img_data = cv2.imread(os.path.join(mask_path, i), 0)
    reshaped = img_data.reshape(-1)
    img_array.append(reshaped)
    counts += np.bincount(reshaped)

reshaped1 = np.array(img_array).reshape(-1)
cls_wtg = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(reshaped1), y=reshaped1)
print("Counts per each class = ", counts)
print("Class weight = ", cls_wtg)

train.TRAIN_IMG_DIR = image_path
train.TRAIN_MSK_DIR = mask_path
train.TEST_IMG_DIR = image_path
train.TEST_MSK_DIR = mask_path
train.CLASS_WEIGHT = t.tensor(cls_wtg, device=DEVICE, dtype=t.float32)
train.main()