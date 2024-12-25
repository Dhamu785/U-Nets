import os
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision import transforms

class seg_dataset(Dataset):
    def __init__(self, path):
        ...