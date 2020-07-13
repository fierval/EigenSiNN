import commondata2d as cd
import torch
import torch.nn as nn

inp = cd.inp
fakeloss = cd.fakeloss

rl = nn.ReLU()
output = rl(inp)
output.backward(fakeloss.reshape_as(output))