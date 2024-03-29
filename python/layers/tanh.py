#%%
import os, sys
from utils import to_cpp

import tstcommon.commondata2d as cd
import torch
import torch.nn as nn

#%%
############# ReLU ############################
inp = cd.inp
fakeloss = cd.fakeloss

sg = nn.Tanh()
output = sg(inp)
output.backward(fakeloss.reshape_as(output))

# %%
