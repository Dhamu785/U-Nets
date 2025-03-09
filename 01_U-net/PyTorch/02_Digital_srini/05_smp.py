# %% import libaries
import segmentation_models_pytorch as smp
import torch as t

# %% checks
print(t.cuda.is_available())
print(t.version.cuda)
print(t.backends.cudnn.version())
print(t.backends.cudnn.enabled)

# %% Declar the variables
NUM_CLASS = 4
DEVICE = 'cuda' if t.cuda.is_available() else 'cpu'
print(f"Available device = {DEVICE}")
pre_trained_encoder = "resnet101"
pre_trained_weight = 'imagenet'

# %% Pre-process images
pre_process = smp.encoders.get_preprocessing_fn(pre_trained_encoder, pre_trained_weight)
# %%Load pre-trained model
model = smp.Unet(encoder_name = pre_trained_encoder, encoder_weights=pre_trained_weight,
                    activation='softmax', classes=NUM_CLASS)

# %% Pass some random values
random = t.randn((1024,1024,3))
prcsd = pre_process(random)
prcsd.shape