# %% import libaries
import segmentation_models_pytorch as smp
# %%
NUM_CLASS = 4

pre_trained_encoder = "densenet121"
pre_trained_weight = 'imagenet'
model = smp.Unet(encoder_name = pre_trained_encoder, encoder_weights=pre_trained_weight,
                    activation='softmax', classes=NUM_CLASS)
# %%
