# %% import lib
import torch as t
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image
from unet import unet

# %% load model and analysis the device
DEVICE = 'cuda' if t.cuda.is_available() else 'cpu'
print(DEVICE)

## method-1
# model = t.load(r"C:\Users\dhamu\Documents\Python all\torch_works\01\model\u-net\v0\entire-model.pt")

## method-2
model = unet(in_channel=3, num_classes=1).to(DEVICE)
model.load_state_dict(t.load(r"C:\Users\dhamu\Documents\Python all\torch_works\01\model\u-net\v0\seg-model.pt", map_location=t.device(DEVICE), weights_only=True))

# print(model.state_dict())
# %%
# type-1
mdl = iter(model.parameters())
print(f"Type-1 of locating the model: {next(mdl).device}")
# type-2
m = list(model.parameters())
print(f"Type-2 of locating the model: {m[0].device}")

# %%
transform_img = transforms.Compose([
            transforms.Resize((512,512)),
            transforms.ToTensor()
        ])
# Original car image
img = Image.open(r"C:\Users\dhamu\Documents\Python all\torch_works\01\dataset\X\0cdf5b5d0ce1_13.jpg")

# image mask
# img = Image.open(r"C:\Users\dhamu\Documents\Python all\torch_works\01\dataset\Y\0cdf5b5d0ce1_01_mask.gif").convert('L')
transformed_img = transform_img(img)
transformed_img1 = transformed_img.unsqueeze(0)
channel_rearranged = transformed_img.permute(1, 2, 0)
print(channel_rearranged.shape)
plt.imshow(channel_rearranged)
plt.axis('off')
plt.show()
# %% model predictions
with t.inference_mode():
    pred = model(transformed_img1.to(DEVICE))

print(pred.shape)
# %%
print(pred.shape)
pred_ch_order = pred.squeeze(0).permute(1, 2, 0)
pred_ch_order_clone = pred_ch_order.clone()
pred_ch_order_clone[pred_ch_order_clone <= 0] = 0
pred_ch_order_clone[pred_ch_order_clone > 0] = 1
print(pred_ch_order_clone.shape)
plt.imshow(pred_ch_order_clone.to('cpu'), cmap='gray')
plt.axis('off')
plt.show()
# %% view model
print(model)
# %% vis model summary
## Type-1
from torchinfo import summary
# %%
model = unet(in_channel=3, num_classes=1).to(DEVICE)
model.load_state_dict(t.load(r"C:\Users\dhamu\Documents\Python all\torch_works\01\model\u-net\v0\seg-model.pt", map_location=t.device(DEVICE), weights_only=True))

summary(model.to('cpu'), input_size=(1, 3, 512, 512))
# %% Type-2
from torchsummary import summary
# %%
summary(model, input_size=(3, 512, 512))
