from sklearn.utils import class_weight
# np.bincount
from torch.utils.data.dataset import Dataset
from torchvision import transforms
import os

class DataSet(Dataset):
    def __init__(self, path_train, path_mask, transforms=None):
        self.path_train = path_train
        self.path_mask = path_mask
        self.transform = transforms
        self.images = os.listdir(self.path_train)
        self.masks = os.listdir(self.path_mask)
        if transforms != None:
            self.transform = transforms
        else:
            self.transform = transforms
    def __len__(self):
        return len(self.masks)
    
    def __getitem__(self, index):
        s_img = os.path.join(self.path_train, self.images[index])
        s_msk = os.path.join(self.path_mask, self.masks[index])