import torch as t
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from torch import nn, optim
from torchvision import transforms
import torch.nn.functional as F
import torchvision.transforms.functional as TF

import matplotlib.pyplot as plt
from PIL import Image
import os
import random
import shutil
import math
from loss import Edge_IoU

from dataset import seg_dataset
from unet import unet

LEARNING_RATE = 1e-4
BATCH_SIZE = 24
EPOCHS = 50

cwd = os.getcwd()
DATA_PATH = os.path.join(cwd, "combined data")
MODEL_SAVE_PATH = os.path.join(cwd, "runs", "models")
IMG_SAVE_PATH = os.path.join(cwd, "runs", "images")
if os.path.exists(MODEL_SAVE_PATH):
    shutil.rmtree(MODEL_SAVE_PATH)
os.makedirs(MODEL_SAVE_PATH)
if os.path.exists(IMG_SAVE_PATH):
    shutil.rmtree(IMG_SAVE_PATH)
os.makedirs(IMG_SAVE_PATH)

DEVICE = 'cuda' if t.cuda.is_available() else 'cpu'

data = seg_dataset(DATA_PATH)
generator = t.Generator().manual_seed(42)

train_dataset, val_dataset = random_split(dataset=data, lengths=(0.8, 0.2), generator=generator)

TRAIN_NO_OF_BATCHES = math.ceil(len(train_dataset)/BATCH_SIZE)
VAL_NO_OF_BATCHES = math.ceil(len(val_dataset)/BATCH_SIZE)


train_dataloader = DataLoader(train_dataset, BATCH_SIZE, True)
val_dataloader = DataLoader(val_dataset, BATCH_SIZE, True)

print(next(iter(train_dataloader))[1].dtype)

model = unet(in_channel=3, num_classes=1).to(DEVICE)
model_path = os.path.join(cwd, 'base model', 'Image_enhancement_sd-18.pt')
model.load_state_dict(t.load(model_path, map_location=t.device(DEVICE), weights_only=True))
optimizer = optim.Adam(params = model.parameters(), lr=LEARNING_RATE)
loss = nn.BCEWithLogitsLoss()

single_img = os.path.join(cwd, "combined data", "X", "103.jpeg")
single_target = os.path.join(cwd, "combined data", "Y", "103.png")
img = Image.open(single_img).convert('RGB')
msk = Image.open(single_target).convert('L')

transforms_pipe = transforms.Compose([
        transforms.Resize((512,512)),
        transforms.ToTensor()
    ])

img_transformed = transforms_pipe(img)
msk_transformed = transforms_pipe(msk)

def plot_img(model, epoch):
    img = img_transformed.unsqueeze(0).to(DEVICE)
    # print(img.shape)
    pred = model(img)
    pred = t.sigmoid(pred)
    pred = t.where(pred <= 0.5, t.ones_like(pred, device=DEVICE), t.zeros_like(y_pred, device=DEVICE))
    plt.figure(figsize=(15,5))
    plt.subplot(1,3,1)
    plt.imshow(img_transformed.permute(1, 2, 0).to('cpu'))
    plt.axis('off')
    plt.title('Original img')
    plt.subplot(1,3,2)
    plt.imshow(msk_transformed[0].to('cpu'), cmap='gray')
    plt.title('Original mask')
    plt.axis('off')
    plt.subplot(1,3,3)
    plt.imshow(pred[0][0].detach().to('cpu').numpy(), cmap='gray')
    plt.title('Predicted mask')
    plt.axis('off')
    plt.savefig(os.path.join(IMG_SAVE_PATH, f"epoch_{epoch}.png"))
    plt.close()

loss_iou = Edge_IoU(device=DEVICE)

scaler = t.GradScaler(device=DEVICE)
for epoch in range(1, EPOCHS+1):
    model.train()
    train_loss_per_batch = 0
    bar = tqdm(range(TRAIN_NO_OF_BATCHES), desc="Batch processing", unit="batchs", colour='GREEN')
    for idx,batch in enumerate(train_dataloader):
        img = batch[0].float().to(DEVICE, non_blocking=True)
        mask = batch[1].float().to(DEVICE, non_blocking=True)

        # 1. Forward pass
        with t.autocast(device_type=DEVICE):
            y_pred = model(img)
            # 2. Calculate the loss
            ls = loss_iou(y_pred, mask)

        train_loss_per_batch += ls.item()

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(ls).backward()
        scaler.step(optimizer=optimizer)
        scaler.update()
        bar.update(1)
        bar.set_postfix(loss = f"{ls.item():.4f}")
    bar.close()
    train_loss_per_batch /= idx+1

    model.eval()
    test_loss_per_batch = 0
    with t.inference_mode():
        bar = tqdm(range(VAL_NO_OF_BATCHES), desc="Batch processing", unit="batchs", colour='RED')
        for idx, batch in enumerate(val_dataloader):
            img = batch[0].float().to(DEVICE, non_blocking=True)
            mask = batch[1].float().to(DEVICE, non_blocking=True)

            y_pred_test = model(img)
            test_ls = loss_iou(y_pred_test, mask)

            test_loss_per_batch += test_ls.item()

            bar.update(1)
            bar.set_postfix(loss = f"{test_ls.item():.4f}")
        bar.close()

        test_loss_per_batch /= idx+1
        plot_img(model, epoch)

    print(f"{epoch} / {EPOCHS} | train_loss = {train_loss_per_batch:.4f} | test_loss = {test_loss_per_batch:.4f}")
    t.save(model.state_dict(), os.path.join(MODEL_SAVE_PATH, f"Image_enhancement_sd-{epoch}.pt"))
# t.save(model, "Image_enhancement-em.pt")