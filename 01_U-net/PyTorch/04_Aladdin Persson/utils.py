# %% import libs
import torch as t
import torchvision
from torch.utils.data import DataLoader
from dataset import carvana

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
    train_ds = carvana(traindir, trainmskdir, train_transforms)
    train_loader = DataLoader(train_ds, batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=True)
    test_ds = carvana(testdir, testmskdir, val_transforms)
    test_loader = DataLoader(test_ds, batch_size, pin_memory=pin_memory, num_workers=num_workers)

    return train_loader, test_loader
# %%
def calc_accuracy(model, loader, device:str | str='cuda'):
    num_correct = 0
    num_pixels = 0
    dice_score = 0

    model.eval()
    with t.inference_mode():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1).to(dtype=t.float16)

            preds = t.sigmoid(model(x))
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += t.numel(preds)
            dice_score += ( 2 * (preds * y).sum() / (preds + y).sum() + 1e-8)

    print(f"Got {num_correct}/{num_pixels} with acc of {num_correct/num_pixels*100:.2f}")
    print(f"Dice score = {dice_score / len(loader):.2f}")
    model.train()

# %%
def save_predictions(model, folder_path, loader, device, epoch):
    model.eval()
    for idx, (x,y) in enumerate(loader):
        x = x.to(device)

        with t.inference_mode():
            preds = t.sigmoid(model(x))
            preds_bin = (preds > 0.5).float()

        torchvision.utils.save_image(preds_bin, f"{folder_path}/epoch-{epoch}_predictioni{idx}.png")
        torchvision.utils.save_image(y.unsqueeze(1), f"{folder_path}/epoch-{epoch}_labels-{idx}.png")
        model.train()