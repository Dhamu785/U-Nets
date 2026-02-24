import os
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision import transforms
import numpy as np

import albumentations as A
from albumentations.pytorch import ToTensorV2

class seg_dataset(Dataset):
    def __init__(self, path, test=False):
        self.path = path
        if test:
            self.images = sorted([os.path.join(path, "Clean", i) for i in os.listdir(os.path.join(path, "Clean"))])
            self.labels = sorted([os.path.join(path, "Noisy", i) for i in os.listdir(os.path.join(path, "Noisy"))])
        else:
            self.images = sorted([os.path.join(path, "Clean", i) for i in os.listdir(os.path.join(path, "Clean"))])
            self.labels = sorted([os.path.join(path, "Noisy", i) for i in os.listdir(os.path.join(path, "Noisy"))])

        self.transforms = A.Compose([
            A.Resize(512, 512),  # cv2.INTER_LINEAR for image
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),              # handles 0/90/180/270
            A.GaussianBlur(blur_limit=3, p=0.2),
            ToTensorV2()
        ], additional_targets={'mask': 'mask'}, is_check_shapes=True)

    def __getitem__(self, index):
        img = np.array(Image.open(self.images[index]).convert('RGB'))
        mask = np.array(Image.open(self.labels[index]).convert('L'))
        aug = self.transforms(image=img, mask=mask)

        return aug['image']/255.0, aug['mask'].unsqueeze(0)/255.0

    
    def __len__(self):
        return len(self.images)
    
if __name__ == "__main__":
    data = seg_dataset(r'C:\Users\dhamu\Documents\Python all\torch_works\01\dataset')
    print(data)
    print(len(data))