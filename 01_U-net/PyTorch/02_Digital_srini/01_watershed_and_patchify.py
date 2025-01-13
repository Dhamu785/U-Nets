# %% import libs
# https://youtu.be/lOZDTDOlqfk?si=4qumQIubzNk-0lSg - instance segmentation
import os
import numpy as np
from PIL import Image
from skimage import measure, color, io
import matplotlib.pyplot as plt
import cv2
import pandas as pd

# %% read image
img_data = cv2.imread(r'/Users/dhamodharan/My-Python/AI-Tutorials/U-Nets/01_U-net/PyTorch/02_Digital_srini/data/seg_img.png')
img_gray = img_data[:,:,0]
plt.imshow(img_gray, cmap='gray')
plt.show()
# %% image binarization 
ret1, thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
print(ret1)
plt.imshow(thresh, cmap='gray')
plt.axis('off')
plt.show()
# %% Morphological operations to remove small noise - opening
#To remove holes we can use closing
kernel = np.ones((3,3),np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations = 1)
plt.imshow(opening, cmap='gray')
plt.axis('off')
plt.show()
# %% Dilation for background
sure_bg = cv2.dilate(opening,kernel,iterations=10)
plt.imshow(sure_bg, cmap='gray')
plt.axis('off')
plt.show()
# %% distance transform to find the midpoint
dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
plt.imshow(dist_transform, cmap='gray')
plt.axis('off')
plt.show()
# %% to detect foreground
ret2, sure_fg = cv2.threshold(dist_transform, 0.2*dist_transform.max(),255,0)
plt.imshow(sure_fg, cmap='gray')
plt.axis('off')
plt.show()
# %% To find unknowns
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg, sure_fg)
plt.imshow(unknown, cmap='gray')
plt.axis('off')
plt.show()
# %% Label numbers
ret3, markers = cv2.connectedComponents(sure_fg)
plt.imshow(markers, cmap='gray')
plt.axis('off')
plt.show()
# %%
markers = markers+10
plt.imshow(markers, cmap='gray')
plt.axis('off')
plt.show()
# %% Place the unknowns in marker
markers[unknown==255] = 0
plt.imshow(markers, cmap='gray')
plt.axis('off')
plt.show()
# %% Detect boundries
markers = cv2.watershed(img_data, markers)
plt.imshow(markers, cmap='gray')
plt.axis('off')
plt.show()
# %% Color the labels
img_data[markers == -1] = [0,200,0] 
img2 = color.label2rgb(markers, bg_label=10, bg_color=(0, 0, 0))

plt.imshow(img2)
plt.axis('off')
plt.show()
# %%
plt.imshow(img_data)
plt.axis('off')
plt.show()
# %%

props = measure.regionprops_table(markers, img_gray, 
                            properties=['label', 'area', 'mean_intensity', 
                            'solidity', 'equivalent_diameter'])

df = pd.DataFrame(props)
df.head()

# https://youtu.be/LM9yisNYfyw?si=yfS69X_scKEQBvDb patch images

# %% image patchify
from patchify import patchify, unpatchify
import cv2
import matplotlib.pyplot as plt
# %%
img_data = cv2.imread(r'/Users/dhamodharan/My-Python/AI-Tutorials/U-Nets/01_U-net/PyTorch/02_Digital_srini/data/seg_img.png')
print(img_data.shape)
img_resized = cv2.resize(img_data, (512, 512))
print(img_resized.shape)
# %%
img_patchs = patchify(img_resized, (64, 64, 3), 64)
print(img_patchs.shape)

#%%
plt.imshow(img_patchs[0,0,0])
plt.axis('off')
plt.show()
# %%
unpatch = unpatchify(img_patchs, (512, 512,3))
print(unpatch.shape)
# %%
