#%%
import os, sys
from utils import to_cpp

import commondata2d as cd
import torch
import torch.nn as nn

#%%
############# ReLU ############################
inp = cd.inp
fakeloss = cd.fakeloss

sg = nn.Sigmoid()
output = sg(inp)
output.backward(fakeloss.reshape_as(output))

# %%
