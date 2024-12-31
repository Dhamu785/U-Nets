import torch as t
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from torch import nn, optim

from dataset import seg_dataset
from unet import unet

if __name__ == "__main__":
    LEARNING_RATE = 1e-4
    BATCH_SIZE = 4
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

    def accuracy(y_pred, y_true):
        y_pred = y_pred.reshape((-1))
        y_true = y_true.reshape((-1))
        count = t.eq(y_pred, y_true).sum().item()
        return (count/len(y_pred)) * 100

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
            ls = loss(y_pred, mask)

            acc = accuracy(y_pred, mask)
            train_loss_per_batch += ls
            train_acc_per_batch += acc

            optimizer.zero_grad()
            ls.backward()
            optimizer.step()
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
                test_ls = loss(y_pred_test, mask)
                test_acc = accuracy(y_pred_test, mask)

                test_loss_per_batch += test_ls
                test_acc_per_batch += test_acc
            
            test_loss_per_batch /= idx+1
            test_acc_per_batch /= idx+1

        print(f"{epoch} / {EPOCHS} | train_loss = {train_loss_per_batch:.4f} | train_acc = {train_acc_per_batch:.4f} | test_loss = {test_loss_per_batch:.4f} | test_acc = {test_acc_per_batch:.4f}")
