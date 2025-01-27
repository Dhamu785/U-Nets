# %% Import libaries
import torch as t
import torch.nn as nn
import torchvision.transforms.functional as TF

# %% Create layers
class DoubleConv(nn.Module):
    def __init__(self, in_channel:int, out_channel:int) -> t.tensor:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels = in_channel, out_channels = out_channel, 
                        kernel_size = 3, stride = 1, padding = 1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace = True),
            nn.Conv2d(in_channels = in_channel, out_channels = out_channel, 
                        kernel_size = 3, stride = 1, padding = 1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace = True)
        )

    def forward(self, x):
        return self.conv(x)

class UNET(nn.Module):
    def __init__(self, in_channel:int | int=3, out_channel:int | int = 1, features = [64, 128, 256, 512]):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)

        # Left part of u-net
        for feature in features:
            self.downs.append(DoubleConv(in_channel = in_channel, out_channel = feature))
            in_channel = feature

        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(feature*2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.output_layer = DoubleConv(features[0], out_channel=out_channel)

    def forwars(self, x):
        skip_connections = []