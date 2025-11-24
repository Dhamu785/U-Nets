import os
from PIL import Image
from torch.utils.data.dataset import Dataset
# from torchvision import transforms

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
                        A.Resize(512, 512),
                        # always flip
                        A.HorizontalFlip(p=1.0),
                        A.VerticalFlip(p=1.0),
                        A.RandomRotate90(p=1.0),
                        ToTensorV2()
                    ], is_check_shapes=False)

    def __getitem__(self, index):
        img = np.array(Image.open(self.images[index]).convert('RGB'))
        mask = np.array(Image.open(self.labels[index]).convert('L'))

        augmented = self.transforms(image=img, mask=mask)

        img = augmented["image"]
        mask = augmented["mask"]

        # return self.transforms(img), self.transforms(mask)
        return img, mask.unsqueeze(0)
    
    def __len__(self):
        return len(self.images)
    
if __name__ == "__main__":
    data = seg_dataset('F:\\CBIR\\data\\combined')
    print(data)
    print(len(data))