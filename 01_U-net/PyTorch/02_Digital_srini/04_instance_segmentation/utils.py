import torch as t
import torchvision
from torch.utils.data import DataLoader
from dataset import DataSet

def save_checkpoint(state, filename="checkpoint.ckpt"):
    # print("==> Saving the checkpoint")
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

def calc_accuracy(model, loader, classes: int, device:str | str='cuda'):
    num_correct = t.zeros((classes), dtype=t.float, device=device)
    num_pixels = t.zeros((classes), dtype=t.float, device=device)
    dice_score = t.zeros((classes), dtype=t.float, device=device)
    model.eval()
    with t.inference_mode():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1).to(dtype=t.int)
            preds = t.argmax(model(x), 1)
            for clss in range(classes):
                p = (preds == clss).int()
                true = (y == clss).int()
                num_correct[clss] += (p == true).sum()
                dice_score[clss] += (2 * (p * true).sum() / (p + true).sum() + 1e-8)
                num_pixels[clss] += true.sum()
        print("Dice score : ", end=' ')
        cl_acc = []
        for score in range(len(dice_score)):
            class_acc = dice_score[score] / len(loader)
            cl_acc.append(class_acc)
            print(f"class_{score} = {class_acc:.2f}", end='\t')
        print(f"Overall accuracy = , {t.mean(t.tensor(cl_acc, dtype=t.float32)).item():.3f}%")
    model.train()


def save_predictions(model, folder_path, loader, device, epoch):
    model.eval()
    for idx, (x,y) in enumerate(loader):
        x = x.to(device)

        with t.inference_mode():
            preds = t.argmax(model(x),1).to(dtype=t.float)

        torchvision.utils.save_image(preds.unsqueeze(1), f"{folder_path}/epoch-{epoch}_predictioni{idx}.png")
        torchvision.utils.save_image(y.unsqueeze(1), f"{folder_path}/epoch-{epoch}_labels-{idx}.png")
        model.train()