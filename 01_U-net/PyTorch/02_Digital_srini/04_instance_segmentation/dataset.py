import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

class Dataset(Dataset):
    def __init__(self, image_path:str, mask_path:str, transform=None):
        self.img_path = image_path
        self.msk_path = mask_path
        self.transform = transform
        self.images = os.listdir(self.img_path)
        self.masks = os.listdir(self.msk_path)

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        s_img = os.path.join(self.img_path, self.images[index])
        s_msk = os.path.join(self.msk_path, self.images[index].replace('.jpg', '_mask.gif'))

        arr_img = np.array(Image.open(s_img).convert('RGB'))
        arr_msk = np.array(Image.open(s_msk).convert('L'), dtype=np.float32)
        arr_msk[arr_msk == 255.0] = 1.0

        if self.transform is not None:
            augmentations = self.transform(image = arr_img, mask = arr_msk)
            image = augmentations['image']
            mask = augmentations['mask']

        return image, mask
