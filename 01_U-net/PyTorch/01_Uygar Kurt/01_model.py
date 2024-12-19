# %% imports and devicesetup

import torch as t
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt

device = 'cuda' if t.cuda.is_available() else 'cpu'
# %%
