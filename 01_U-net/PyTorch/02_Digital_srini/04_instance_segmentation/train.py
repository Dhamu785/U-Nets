# %% import libs
import torch as t
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model import UNET
from utils import save_checkpoint, load_checkpoint, get_loaders, calc_accuracy, save_predictions

# %% Hyperparameter
LEARNING_RATE = 1e-4
DEVICE = 'cuda' if t.cuda.is_available() else 'cpu'
print(f"Available device = {DEVICE}")
BATCH_SIZE = 8
NUM_EPOCHS = 10
NUM_WORKERS = 0
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
PIN_MEMORY = True
LOAD_MODEL = False
TRAIN_IMG_DIR = ''
TRAIN_MSK_DIR = ''
TEST_IMG_DIR = ""
TEST_MSK_DIR = ""
CLASS_WEIGHT = []

# %% train
def train_fn(loader, model, optimizer, loss_fn, scalar):
    loop = tqdm(loader)

    for batch_idx, (images, targets) in enumerate(loop):
        images = images.to(device=DEVICE)
        targets = targets.to(device=DEVICE, dtype=t.int64)

        # forward pass & loss calculations
        with t.autocast(device_type=DEVICE):
            predictions = model(images)
            loss = loss_fn(predictions.to(t.float32), targets)

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

    model = UNET(in_channel=3, out_channel=4).to(device=DEVICE)
    # loss_fn = nn.BCEWithLogitsLoss()
    loss_fn = nn.CrossEntropyLoss(weight=CLASS_WEIGHT)
    optimizer = optim.Adam(params=model.parameters(), lr=LEARNING_RATE)
    
    train_loader, val_loader = get_loaders(TRAIN_IMG_DIR, TRAIN_MSK_DIR, TEST_IMG_DIR, TEST_MSK_DIR,
                                            BATCH_SIZE, train_transform, val_transform, NUM_WORKERS, PIN_MEMORY)
    
    if LOAD_MODEL:
        load_checkpoint(model, "checkpoint.ckpt")

    calc_accuracy(model, val_loader, 4, DEVICE)

    scaler = t.GradScaler(DEVICE)

    for epoch in range(NUM_EPOCHS):
        train_fn(train_loader, model, optimizer, loss_fn, scaler)

        check_point = {"state_dict": model.state_dict(), optimizer:optimizer.state_dict()}
        save_checkpoint(check_point)

        calc_accuracy(model, val_loader, 4, DEVICE)

        save_predictions(model, "image_saved_per_epoch", val_loader, DEVICE, epoch)

if __name__ == "__main__":
    main()