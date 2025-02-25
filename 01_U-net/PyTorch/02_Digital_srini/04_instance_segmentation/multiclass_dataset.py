from sklearn.utils import class_weight
# np.bincount
from torch.utils.data.dataset import Dataset
from torchvision import transforms
import os
from PIL import Image
import torch as t
import numpy as np

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
        # numpy_img = np.array(s_msk)
        # h,w = numpy_img.shape
        # s_mask_reshaped = numpy_img.reshape((-1))
        # onehot_msk = t.nn.functional.one_hot(t.tensor(s_mask_reshaped, dtype=t.long), 4)
        # original_shape = onehot_msk.reshape(h, w, -1)
        # return self.transform(s_img), np.transpose(original_shape, (2, 0, 1))
        return self.transform(s_img), self.transform(s_msk)


if __name__ == "__main__":
    path_img = '/Users/dhamodharan/My-Python/AI-Tutorials/dataset/Types of seg/02_Multi/X'
    path_msk = '/Users/dhamodharan/My-Python/AI-Tutorials/dataset/Types of seg/02_Multi/Y'
    data = DataSet(path_img, path_msk)
    print(f"Options available for Dataset = {dir(data)}")
    print(f"Shape od the mask image = {data[1][1].shape}")
    print(f"Length of the data = {len(data)}")