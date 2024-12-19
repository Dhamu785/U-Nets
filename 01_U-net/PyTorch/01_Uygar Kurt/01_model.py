# %% imports and devicesetup

import torch as t
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn

device = 'cuda' if t.cuda.is_available() else 'cpu'
# %%
 

class DoubleConv:
    def __init__(self, in_channel, out_channels):
        super().__init__()
        self.conv_op = nn.Sequential(
            nn.Conv2d(in_channel, out_channels, kernel_size=(3,3), padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3,3), padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv_op(x)