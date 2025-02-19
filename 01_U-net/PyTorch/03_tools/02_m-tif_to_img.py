# %% import libs
from PIL import Image 

# %% init path
# Path for windows
# target_path_msk = "C:\\Users\\dhamu\\Documents\\Python all\\torch_works\\01\\00_datasets-from_online\\01_Segmentations\\02_Instance\\02\\sandstone_data_for_ML\\data_for_3D_Unet\\train_masks"
# tiff_msk = "C:\\Users\\dhamu\\Documents\\Python all\\torch_works\\01\\00_datasets-from_online\\01_Segmentations\\02_Instance\\02\\sandstone_data_for_ML\\data_for_3D_Unet\\train_masks_256_256_256.tif"
# tiff_img = "C:\\Users\\dhamu\\Documents\\Python all\\torch_works\\01\\00_datasets-from_online\\01_Segmentations\\02_Instance\\02\\sandstone_data_for_ML\\data_for_3D_Unet\\train_images_256_256_256.tif"
# target_path_img = "C:\\Users\\dhamu\\Documents\\Python all\\torch_works\\01\\00_datasets-from_online\\01_Segmentations\\02_Instance\\02\\sandstone_data_for_ML\\data_for_3D_Unet\\train_img"

# Path for mac
target_path_msk = '/Users/dhamodharan/My-Python/AI-Tutorials/01_Dataset from online/sandstone_data_for_ML/01_Segmentation/02_Instance segmentation/data_for_3D_Unet/target_masks/'
tiff_msk = '/Users/dhamodharan/My-Python/AI-Tutorials/01_Dataset from online/sandstone_data_for_ML/01_Segmentation/02_Instance segmentation/data_for_3D_Unet/train_images_256_256_256.tif'

# %%
tiff_img = Image.open(tiff_msk)

page_num = 0
while True:
    try:
        tiff_img.seek(page_num)
        tiff = tiff_img.convert('L')
        tiff_img.save(f"{target_path_msk}/img_{page_num}.png")
        page_num += 1
    except Exception as e:
        print(e)
        break