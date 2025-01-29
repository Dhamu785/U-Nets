# %% import libs
import torch as t
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model import UNET

# %% Hyperparameter
LEARNING_RATE = 1e-4
DEVICE = 'cuda' if t.cuda.is_available() else 'cpu'
BATCH_SIZE = 16
NUM_EPOCHS = 20
NUM_WORKERS = 2
IMAGE_HEIGHT = 160
IMAGE_WIDTH = 240
PIM_MEMORY = True
LOAD_MODEL = False
TRAIN_IMG_DIR = ''
TRAIN_MSK_DIR = ''
TEST_IMG_DIR = ''
TEST_MSK_DIR = ''

# %% train
def train_fn(loader, model, optimizer, loss_fn, scalar):
    loop = tqdm(loader)

    for batch_idx, (images, targets) in enumerate(loop):
        images = images.to(device=DEVICE)
        targets = targets.unsqueeze(1).float().to(device=DEVICE)

        # forward pass & loss calculations
        with t.autocast(device_type=DEVICE):
            predictions = model(images)
            loss = loss_fn(predictions, targets)

        # zero-grad and backward pass
        optimizer.zero_grad()
        scalar.scale(loss).backward()

        # update optimizers
        scalar.step(optimizer)
        scalar.update()

        loop.set_postfix(loss=loss.item())

# %% main function
def main():
    train_transform = A.Compose([
        A.Resize(height = IMAGE_HEIGHT, width = IMAGE_WIDTH),
        A.Rotate(limit=35, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
        ToTensorV2()
    ])

    val_transform = A.Compose([
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
        ToTensorV2()
    ])

    model = UNET(in_channel=3, out_channel=1).to(device=DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(params=model.parameters(), lr=LEARNING_RATE)
    