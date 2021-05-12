import os
from utils import to_cpp
import commondata4d as cd
import torch.nn as nn

############ ReLU ############################
inp = cd.inp_with_neg
fakeloss = cd.convloss

rl = nn.ReLU()
output = rl(inp)
output.backward(cd.inp)
print(f"RELU output: {to_cpp(output)}")
print(f"RELU grad: {to_cpp(inp.grad.data)}")

############# Leaky ReLU #####################
inp.grad.zero_()
lrl = nn.LeakyReLU(negative_slope=1e-2)
output = lrl(inp)
output.backward(cd.inp)

print(f"LeakyRELU output: {to_cpp(output)}")
print(f"LeakyRELU grad: {to_cpp(inp.grad.data)}")
