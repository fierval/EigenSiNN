#%%
import os, sys
from utils import to_cpp

tstpath = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, tstpath)

import tstcommon.commondata2d as cd
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
