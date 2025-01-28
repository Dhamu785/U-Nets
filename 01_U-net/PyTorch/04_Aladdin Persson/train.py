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
        targets = targets.float().to(device=DEVICE)

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