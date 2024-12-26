import os
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision import transforms

class seg_dataset(Dataset):
    def __init__(self, path, test=False):
        self.path = path
        if test:
            self.images = sorted([os.path.join(path, "X", i) for i in os.listdir(os.path.join(path, "X"))])
            self.labels = sorted([os.path.join(path, "Y", i) for i in os.listdir(os.path.join(path, "Y"))])
        else:
            self.images = sorted([os.path.join(path, "X", i) for i in os.listdir(os.path.join(path, "X"))])
            self.labels = sorted([os.path.join(path, "Y", i) for i in os.listdir(os.path.join(path, "Y"))])

        self.transforms = transforms.Compose([
            transforms.Resize((512,512)),
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        mask = Image.open(self.labels[index]).convert('L')

        return self.transforms(img), self.transforms(mask)
    
    def __len__(self):
        return len(self.images)
    
if __name__ == "__main__":
    data = seg_dataset(r'C:\Users\dhamu\Documents\Python all\torch_works\01\dataset')
    print(data)
    print(len(data))