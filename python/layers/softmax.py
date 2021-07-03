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

sg = nn.Softmax(dim=2)
output = sg(inp)
output.backward(fakeloss.reshape_as(output))

print(f"output: {to_cpp(output)}")
print(f"dinput: {to_cpp(inp.grad)}")
# %%
