import commondata2d as cd
import torch
import torch.nn as nn

############# ReLU ############################
inp = cd.inp
fakeloss = cd.fakeloss

rl = nn.ReLU()
output = rl(inp)
output.backward(fakeloss.reshape_as(output))

############# Leaky ReLU #####################
inp.grad.zero_()
lrl = nn.LeakyReLU(negative_slope=1e-2)
output = lrl(inp)
output.backward(fakeloss.reshape_as(output))