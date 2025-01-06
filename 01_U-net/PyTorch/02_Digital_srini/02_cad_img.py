# %% import libs
import numpy as np
from skimage import color, measure
import matplotlib.pyplot as plt
import cv2
# %% reading and ploting the images
img_data = cv2.imread(r'C:\Users\dhamu\Documents\Python all\torch_works\01\U-Nets\01_U-net\PyTorch\02_Digital_srini\data\30450.png')
gray_img = img_data[:,:,0]

plt.subplot(1, 2, 1)
plt.imshow(img_data)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(gray_img, cmap='gray')
plt.axis('off')
plt.show()
# %% Image binarixation
ret1, thresh = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
print(ret1)
# thresh = cv2.bitwise_not(thresh)
plt.imshow(thresh, cmap='gray')
plt.axis('off')
plt.show()
# %%
kernel = np.ones((3,3),np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations = 5)
plt.imshow(opening, cmap='gray')
plt.axis('off')
plt.show()
# %%
sure_bg = cv2.dilate(opening, kernel, iterations=1)
plt.imshow(sure_bg, cmap='gray')
plt.axis('off')
plt.show()
# %%
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
plt.imshow(dist_transform, cmap='gray')
plt.axis('off')
plt.show()
# %%
ret2, sure_fg = cv2.threshold(dist_transform, 0.1*dist_transform.max(),255,0)
plt.imshow(sure_fg, cmap='gray')
plt.axis('off')
plt.show()
# %%
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg, sure_fg)
plt.imshow(unknown, cmap='gray')
plt.axis('off')
plt.show()
# %%
ret3, markers = cv2.connectedComponents(sure_fg)
plt.imshow(markers, cmap='gray')
plt.axis('off')
plt.show()
# %%
markers = markers+10
plt.imshow(markers, cmap='gray')
plt.axis('off')
plt.show()
# %%
markers[unknown == 255] = 0
plt.imshow(markers, cmap='gray')
plt.axis('off')
plt.show()
# %%
markers = cv2.watershed(img_data, markers)
plt.imshow(markers, cmap='gray')
plt.axis('off')
plt.show()
# %%
img_data[markers == -1] = [0,200,0] 
img2 = color.label2rgb(markers, bg_label=10, bg_color=(0, 0, 0))

plt.imshow(img2)
plt.axis('off')
plt.show()