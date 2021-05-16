#%%
import os, sys
from utils import to_cpp

import commondata4d as cd
import torch
import torch.nn as nn

#%%
############# ReLU ############################
inp = cd.inp
fakeloss = cd.inp

sg = nn.Sigmoid()
output = sg(inp)
output.backward(fakeloss)

# %%
print(f"Sigmoid: {to_cpp(output)}")
print(f"Sigmoid Grad: {to_cpp(inp.grad.data)}")