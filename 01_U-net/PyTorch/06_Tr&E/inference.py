import torch as t
from torchvision import transforms
from PIL import Image
from unet import unet
import os
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import shutil

DEVICE = 'cuda' if t.cuda.is_available() else 'cpu'
cwd = os.getcwd()
PATH = os.path.join(cwd, 'data', 'processed')
SAV_DIR = os.path.join(cwd, 'data', 'enhanced')
if os.path.exists(SAV_DIR):
    shutil.rmtree(SAV_DIR)
os.makedirs(SAV_DIR)

model = unet(in_channel=3, num_classes=1).to(DEVICE)
model.load_state_dict(t.load(os.path.join(cwd, "models", "Image_enhancement_sd.pt"), map_location=t.device(DEVICE), weights_only=True))
transform_img = transforms.Compose([
                transforms.Resize((512,512)),
                transforms.ToTensor()
            ])

def predict_and_save(src_path, model, transform, save_path):
    global DEVICE
    imgs = os.listdir(src_path)
    for i in tqdm(range(len(imgs))):
        img = Image.open(os.path.join(src_path, imgs[i]))
        img = transform(img)
        img = img.unsqueeze(0)
        with t.inference_mode():
            pred_img = model(img.to(DEVICE))
            act_img = F.sigmoid(pred_img)
        bi_img = t.where(act_img <= 0.5, t.ones_like(pred_img, device=DEVICE),
                        t.zeros_like(pred_img, device=DEVICE))
        re_arrange_img = bi_img.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
        # print(re_arrange_img.shape)
        im = Image.fromarray((re_arrange_img.squeeze() * 255).astype(np.uint8))
        im.save(os.path.join(save_path, imgs[i].split('.')[0]+'.png'))
        
if __name__ == "__main__":
    predict_and_save(PATH, model, transform_img, SAV_DIR)