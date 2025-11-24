import os
import torch as t
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from torch import nn, optim
from torchvision import transforms
import torch.nn.functional as F

import matplotlib.pyplot as plt
from PIL import Image
import shutil

from dataset import seg_dataset
from unet import unet

sav_mdl = 'models'
if os.path.exists(sav_mdl):
    shutil.rmtree(sav_mdl)
os.mkdir(sav_mdl)

def train(lr: float, bth_size: int, epoch:int, data_path: str, sample_x: str, sample_y: str, pre_trained: bool, model_path: str=None):
    LEARNING_RATE = lr
    BATCH_SIZE = bth_size
    EPOCHS = epoch
    DATA_PATH = data_path
    MODEL_SAVE_PATH = "models"
    DEVICE = 'cuda' if t.cuda.is_available() else 'cpu'

    data = seg_dataset(DATA_PATH)
    generator = t.Generator().manual_seed(42)

    train_dataset, val_dataset = random_split(dataset=data, lengths=(0.8, 0.2), generator=generator)
    
    train_dataloader = DataLoader(train_dataset, BATCH_SIZE, True)
    val_dataloader = DataLoader(val_dataset, BATCH_SIZE, True)

    model = unet(in_channel=3, num_classes=1).to(DEVICE)
    if pre_trained:
        model.load_state_dict(t.load(model_path, map_location=t.device(DEVICE), weights_only=True))
    optimizer = optim.Adam(params = model.parameters(), lr=LEARNING_RATE)
    # loss = nn.BCEWithLogitsLoss()

    single_img = sample_x
    single_target = sample_y
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
        pred = F.sigmoid(pred)
        pred = t.where(pred <= 0.5, t.ones_like(y_pred, device=DEVICE), t.zeros_like(y_pred, device=DEVICE))
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
        # plt.savefig(f"epoch_{epoch}.png")
        plt.show()

    def loss_iou(y_pred, y_true, inf):
        if not inf:
            if not y_pred.requires_grad:
                raise ValueError("y_pred should have gradient tracking")
    
        device = y_pred.device
        y_true = t.where(y_true <= 0, t.ones_like(y_pred, device=device), t.zeros_like(y_pred, device=device))
        y_pred = F.sigmoid(y_pred)
        # y_pred = t.where(y_pred <= 0.5, t.zeros_like(y_pred, device=device), t.ones_like(y_pred, device=device), requires_grad=True)
        
        intersection = t.abs((y_pred.view((-1)) * y_true.view((-1))).sum().float())
        union = t.abs((y_pred.sum() + y_true.sum()).float())
        iou = (t.abs(intersection) + 1e-5) / ((union + 1e-5) - t.abs(intersection))
        iou_loss = 1 - iou
        return iou_loss

    def acc_iou(y_pred, y_true, inf):
        ls = loss_iou(y_pred, y_true, inf)
        return 1-ls

    def accuracy(y_pred, y_true):
        y_pred = y_pred.reshape((-1))
        y_true = y_true.reshape((-1))
        count = t.eq(y_pred, y_true).sum().item()
        return (count/len(y_pred)) * 100

    train_ls = []
    train_acc = []
    val_ls = []
    val_acc = []

    for epoch in range(EPOCHS):
        model.train()
        train_loss_per_batch = 0
        train_acc_per_batch = 0
        epoch_pbar = tqdm(range(len(train_dataloader)), desc="Batch processing",unit="batchs")
        for idx,batch in enumerate(train_dataloader):
            img = batch[0].float().to(DEVICE)
            mask = batch[1].float().to(DEVICE)

            # 1. Forward pass
            y_pred = model(img)
            # 2. Calculate the loss
            ls = loss_iou(y_pred, mask, False)

            acc = acc_iou(y_pred, mask, False)
            train_loss_per_batch += ls.item()
            train_acc_per_batch += acc.item()

            optimizer.zero_grad()
            ls.backward()
            optimizer.step()
            epoch_pbar.set_postfix(loss=ls.item(), accuracy=acc.item())
            epoch_pbar.update(1)
        epoch_pbar.close()
        train_loss_per_batch /= idx+1
        train_acc_per_batch /= idx+1

        model.eval()
        test_loss_per_batch = 0
        test_acc_per_batch = 0
        with t.inference_mode():
            for idx, batch in enumerate(val_dataloader):
                img = batch[0].float().to(DEVICE)
                mask = batch[1].float().to(DEVICE)

                y_pred_test = model(img)
                test_ls = loss_iou(y_pred_test, mask, True)
                test_acc = acc_iou(y_pred_test, mask, True)

                test_loss_per_batch += test_ls.item()
                test_acc_per_batch += test_acc.item()
            
            test_loss_per_batch /= idx+1
            test_acc_per_batch /= idx+1
            plot_img(model, epoch)

        train_ls.append(train_loss_per_batch)
        train_acc.append(train_acc_per_batch)
        val_ls.append(test_loss_per_batch)
        val_acc.append(test_acc_per_batch)

        print(f"{epoch} / {EPOCHS} | train_loss = {train_loss_per_batch:.4f} | train_acc = {train_acc_per_batch:.4f} | test_loss = {test_loss_per_batch:.4f} | test_acc = {test_acc_per_batch:.4f}")
        t.save(model.state_dict(), os.path.join(os.getcwd(), MODEL_SAVE_PATH, f"Image_enhancement_sd-{epoch}.pt"))
    # t.save(model, "Image_enhancement-em.pt")
    return train_ls, train_acc, val_ls, val_acc