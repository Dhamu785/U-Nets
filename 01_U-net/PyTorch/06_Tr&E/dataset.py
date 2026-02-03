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
            self.images = sorted([os.path.join(path, "X", i) for i in os.listdir(os.path.join(path, "X"))])
            self.labels = sorted([os.path.join(path, "Y", i) for i in os.listdir(os.path.join(path, "Y"))])
        else:
            self.images = sorted([os.path.join(path, "X", i) for i in os.listdir(os.path.join(path, "X"))])
            self.labels = sorted([os.path.join(path, "Y", i) for i in os.listdir(os.path.join(path, "Y"))])

        # self.transforms = transforms.Compose([
        #     transforms.Resize((512,512)),
        #     transforms.ToTensor()
        # ])

        self.transforms = A.Compose([
            A.Resize(512, 512, interpolation=1),  # cv2.INTER_LINEAR for image
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),              # handles 0/90/180/270
            A.GaussianBlur(blur_limit=(5, 5), sigma_limit=(1.5, 1.5), p=0.5),  # image-only
            A.Resize(512, 512, interpolation=1),
            ToTensorV2()
        ], additional_targets={'mask': 'mask'}, is_check_shapes=False)

    def __getitem__(self, index):
        img = np.array(Image.open(self.images[index]).convert('RGB'))
        mask = np.array(Image.open(self.labels[index]).convert('L'))
        aug = self.transforms(image=img, mask=mask)
        # img = Image.open(self.images[index]).convert('RGB')
        # mask = Image.open(self.labels[index]).convert('L')

        return aug['image']/255.0, aug['mask'].unsqueeze(0)/255.0
        # return self.transforms(img), self.transforms(mask)

    
    def __len__(self):
        return len(self.images)
    
if __name__ == "__main__":
    data = seg_dataset(r'C:\Users\dhamu\Documents\Python all\torch_works\01\dataset')
    print(data)
    print(len(data))