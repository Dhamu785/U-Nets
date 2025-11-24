import torch as t
import torch.nn as nn

from unet_parts import DoubleConv, Downsample, UpSample

class unet(nn.Module):
    def __init__(self, in_channel, num_classes):
        super().__init__()
        self.down_conv1 = Downsample(in_channel=in_channel, out_channel=64)
        self.down_conv2 = Downsample(in_channel=64, out_channel=128)
        self.down_conv3 = Downsample(in_channel=128, out_channel=256)
        self.down_conv4 = Downsample(in_channel=256, out_channel=512)

        self.bottble_neck = DoubleConv(in_channel=512, out_channels=1024)

        self.up_conv1 = UpSample(in_channel=1024, out_channel=512)
        self.up_conv2 = UpSample(in_channel=512, out_channel=256)
        self.up_conv3 = UpSample(in_channel=256, out_channel=128)
        self.up_conv4 = UpSample(in_channel=128, out_channel=64)

        self.out = nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=(1,1))
        # self.act = nn.Sigmoid()

    def forward(self, x):
        conv1, p1 = self.down_conv1(x)
        conv2, p2 = self.down_conv2(p1)
        conv3, p3 = self.down_conv3(p2)
        conv4, p4 = self.down_conv4(p3)
        
        btl_nk = self.bottble_neck(p4)

        up1 = self.up_conv1(btl_nk, conv4)
        up2 = self.up_conv2(up1, conv3)
        up3 = self.up_conv3(up2, conv2)
        up4 = self.up_conv4(up3, conv1)

        out = self.out(up4)
        # act_out = self.act(out)

        return out
    
if __name__ == "__main__":
    input_img = t.randn((1, 3, 512, 512))
    print(f"Input image shape = {input_img.shape}")

    unet = unet(in_channel=3, num_classes=1)
    res = unet(input_img)
    print(f"Output shape from unet model = {res.shape}")