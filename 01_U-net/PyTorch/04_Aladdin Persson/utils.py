# %% import libs
import torch as t
import torchvision
from torch.utils.data import DataLoader
from dataset import carvana

# %%
def save_checkpoint(state, filename="checkpoint.ckpt"):
    print("==> Saving the checking")
    t.save(state, filename)

def load_checkpoint(model, checkpoint):
    print("==> Loading the model")
    model.load_statedict(checkpoint['state_dict'])

def get_loaders(traindir, trainmskdir, testdir, testmskdir, batch_size, train_transforms,
                val_transforms, num_workers=0, pin_memory=True):
    train_ds = carvana(traindir, trainmskdir, train_transforms)
    train_loader = DataLoader(train_ds, batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=True)
    test_ds = carvana(testdir, testmskdir, val_transforms)
    test_loader = DataLoader(test_ds, batch_size, pin_memory=pin_memory, num_workers=num_workers)

    return train_loader, test_loader
# %%
