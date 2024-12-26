import torch as t
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from torch import nn, optim

from dataset import seg_dataset
from unet import unet

if __name__ == "__main__":
    LEARNING_RATE = 1e-4
    BATCH_SIZE = 16
    EPOCHS = 10
    DATA_PATH = "C:\\Users\\dhamu\\Documents\\Python all\\torch_works\\01\\dataset\\"
    MODEL_SAVE_PATH = "C:\\Users\\dhamu\\Documents\\Python all\\torch_works\\01\\model\\"
    DEVICE = 'cuda' if t.cuda.is_available() else 'cpu'

    data = seg_dataset(DATA_PATH)
    generator = t.Generator().manual_seed(42)

    train_dataset, val_dataset = random_split(dataset=data, lengths=(0.8, 0.2), generator=generator)
    
    train_dataloader = DataLoader(train_dataset, BATCH_SIZE, True)
    val_dataloader = DataLoader(val_dataset, BATCH_SIZE, True)

    model = unet(in_channel=3, num_classes=1).to(DEVICE)
    optimizer = optim.Adam(params = model.parameters(), lr=LEARNING_RATE)
    loss = nn.BCEWithLogitsLoss()