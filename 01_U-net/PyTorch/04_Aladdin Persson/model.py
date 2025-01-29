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
            nn.Conv2d(in_channels = out_channel, out_channels = out_channel, 
                        kernel_size = 3, stride = 1, padding = 1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace = True)
        )

    def forward(self, x):
        return self.conv(x)

class UNET(nn.Module):
    def __init__(self, in_channel:int | int=1, out_channel:int | int = 1, features = [64, 128, 256, 512]):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)

        # Left part of u-net
        for feature in features:
            self.downs.append(DoubleConv(in_channel = in_channel, out_channel = feature))
            in_channel = feature

        # Right part of u-net
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(feature*2, feature))

        # Bottom
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        
        # output
        self.output_layer = nn.Conv2d(in_channels=features[0], out_channels=out_channel, kernel_size = 1)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size = skip_connection.shape[2:])

            x = t.cat((skip_connection, x), dim = 1)
            x = self.ups[idx+1](x)
        
        x = self.output_layer(x)

        return x
# %% test-code
def test():
    sample = t.randn((16, 1, 256, 256))
    print(sample.shape)
    model = UNET(in_channel=1, out_channel=1)
    pred = model(sample)
    print(f"Prediction shape = {pred.shape}, Sample shape = {sample.shape}")
    assert pred.shape == sample.shape

# %% checkpoint test
def checkpoint():
    model = UNET(in_channel=3, out_channel=1)
    model_stat = model.state_dict()
    save_params = {'epochs':20, 'model_state': model_stat}
    t.save(save_params, "files.ckpt")
# %% sample run
if __name__ == "__main__":
    # test()
    checkpoint()

