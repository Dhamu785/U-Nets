# %% import libaries
import segmentation_models_pytorch as smp
import torch as t
import cv2
import os

# %% checks
print(t.cuda.is_available())
print(t.version.cuda)
print(t.backends.cudnn.version())
print(t.backends.cudnn.enabled)

# %% Declar the variables
NUM_CLASS = 1
DEVICE = 'cuda' if t.cuda.is_available() else 'cpu'
print(f"Available device = {DEVICE}")
pre_trained_encoder = "resnet101"
pre_trained_weight = 'imagenet'

# %% Pre-process images
pre_process = smp.encoders.get_preprocessing_fn(pre_trained_encoder, pre_trained_weight)
img_path = '/Users/dhamodharan/My-Python/AI-Tutorials/dataset/Types of seg/01_Binary/X'
img_lst = os.listdir(img_path)
img_data = cv2.imread(os.path.join(img_path, img_lst[10]))
print("Image shape before processing = ", img_data.shape)
prcsd = pre_process(img_data)
print("Image shape after processing = ", prcsd.shape)

# %%Load pre-trained model
model = smp.Unet(encoder_name = pre_trained_encoder, encoder_weights=pre_trained_weight,
                    activation='softmax', classes=NUM_CLASS)

# %% Dummy loss calculations
temp1 = t.tensor([[1,1,1,0,0]], dtype=t.float32)
temp2 = t.tensor([[0.9,0.7,0.8,0,0]], dtype=t.float32)
ls_fn = smp.losses.JaccardLoss('binary', from_logits=True)
loss = ls_fn(temp1, temp2)
print(loss)
# %%
inter = (temp1 * temp2).sum()
union = temp1.sum() + temp2.sum()
ls = inter/(union-inter)
print(f"{inter = }, {union = }, {ls =}")
# %%
t.save(model.state_dict(), 'unet.pt')