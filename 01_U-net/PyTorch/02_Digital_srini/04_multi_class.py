from sklearn.utils import class_weight
# np.bincount
from torch.utils.data.dataset import Dataset
from torchvision import transforms

class DataSet(Dataset):
    def __init__(self, path_train, path_test):
        self.path_train = path_train
        self.path_test = path_test