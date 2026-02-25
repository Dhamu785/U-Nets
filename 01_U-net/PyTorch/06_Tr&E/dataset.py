import os
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision import transforms
import numpy as np

import albumentations as A
from albumentations.pytorch import ToTensorV2

class seg_dataset(Dataset):
    def __init__(self, path):
        self.path = path
        self.images = sorted([os.path.join(path, "Noisy", i) for i in os.listdir(os.path.join(path, "Noisy"))])
        self.labels = sorted([os.path.join(path, "Clean", i) for i in os.listdir(os.path.join(path, "Clean"))])

        self.transforms = A.Compose([
            A.Resize(512, 512),
            A.Affine(translate_percent=0.02, scale=(0.98, 1.02), rotate=(-2, 2), border_mode=0, p=0.5),
            A.GaussNoise(std_range=(0.01, 0.05), p=0.3),
            A.GaussianBlur(blur_limit=3, p=0.2),
            ToTensorV2()
        ], additional_targets={'mask': 'mask'}, is_check_shapes=True)

    def __getitem__(self, index):
        img = np.array(Image.open(self.images[index]).convert('L'), dtype=np.uint8)
        mask = np.array(Image.open(self.labels[index]).convert('L'), dtype=np.uint8)
        aug = self.transforms(image=img, mask=mask)

        return aug['image']/255.0, aug['mask'].unsqueeze(0)/255.0

    
    def __len__(self):
        return len(self.images)
    
if __name__ == "__main__":
    data = seg_dataset(r'C:\Users\dhamu\Documents\Python all\torch_works\01\dataset')
    print(data)
    print(len(data))