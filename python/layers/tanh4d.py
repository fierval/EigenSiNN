#%%
import os, sys
from utils import to_cpp

import tstcommon.commondata4d as cd
import torch
import torch.nn as nn

#%%
############# ReLU ############################
inp = cd.inp
fakeloss = cd.inp

sg = nn.Tanh()
output = sg(inp)
output.backward(fakeloss.reshape_as(output))

# %%
