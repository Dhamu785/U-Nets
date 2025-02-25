# %% import libs
import torch as t
import torchvision
from torch.utils.data import DataLoader
from dataset import DataSet

# %%
def save_checkpoint(state, filename="checkpoint.ckpt"):
    print("==> Saving the checkpoint")
    t.save(state, filename)

def load_checkpoint(model, checkpoint):
    print("==> Loading the model")
    ckpt = t.load(checkpoint, weights_only=False)
    model.load_state_dict(ckpt['state_dict'])

def get_loaders(traindir, trainmskdir, testdir, testmskdir, batch_size, train_transforms,
                val_transforms, num_workers=0, pin_memory=True):
    train_ds = DataSet(traindir, trainmskdir, train_transforms)
    train_loader = DataLoader(train_ds, batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=True)
    test_ds = DataSet(testdir, testmskdir, val_transforms)
    test_loader = DataLoader(test_ds, batch_size, pin_memory=pin_memory, num_workers=num_workers)

    return train_loader, test_loader
# %%
def calc_accuracy(model, loader, classes: int, device:str | str='cuda'):
    # num_correct = t.zeros((classes))
    # num_pixels = t.zeros((classes), dtype=t.float)
    # dice_score = t.zeros((classes), dtype=t.float)

    model.eval()
    with t.inference_mode():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1).to(dtype=t.int)
            preds = t.argmax(model(x), 1)
            for clss in range(classes):
                p = (preds == clss).int()
                t = (y == clss).int()
                num_correct[clss] += (p == t).sum()
                dice_score[clss] += (2 * (p * y).sum() / (p + t).sum() + 1e-8)
                num_pixels[clss] += t.sum()

        for score in range(len(dice_score)):
            print(f"Dice score for class_{score} = {dice_score[score] / len(loader):.2f}")
    model.train()

# %%
def save_predictions(model, folder_path, loader, device, epoch):
    model.eval()
    for idx, (x,y) in enumerate(loader):
        x = x.to(device)

        with t.inference_mode():
            preds = t.argmax(model(x),1)

        torchvision.utils.save_image(preds, f"{folder_path}/epoch-{epoch}_predictioni{idx}.png")
        torchvision.utils.save_image(y.unsqueeze(1), f"{folder_path}/epoch-{epoch}_labels-{idx}.png")
        model.train()