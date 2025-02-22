from sklearn.utils import class_weight
# np.bincount
from torch.utils.data.dataset import Dataset
from torchvision import transforms
import os
from PIL import Image

class DataSet(Dataset):
    def __init__(self, path_img, path_mask, cstm_transform=None):
        self.path_train = path_img
        self.path_mask = path_mask
        self.images = os.listdir(self.path_train)
        self.masks = os.listdir(self.path_mask)
        if cstm_transform == None:
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor()
            ])
        else:
            self.transform = cstm_transform

    def __len__(self):
        return len(self.masks)
    
    def __getitem__(self, index):
        s_img = Image.open(os.path.join(self.path_train, self.images[index])).convert('RGB')
        s_msk = Image.open(os.path.join(self.path_mask, self.masks[index]))
        return self.transform(s_img), self.transform(s_msk)


if __name__ == "__main__":
    path_img = '/Users/dhamodharan/My-Python/AI-Tutorials/dataset/Types of seg/02_Multi/X'
    path_msk = '/Users/dhamodharan/My-Python/AI-Tutorials/dataset/Types of seg/02_Multi/Y'
    data = DataSet(path_img, path_msk)
    print(dir(data))
    print(data[1][1])
    print(len(data))