import sys
import os

sys.path.insert(0, os.path.abspath("../tstcommon"))

import commondata2d as cd

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from utils import to_cpp

inp = cd.inp.reshape(cd.inp.shape[1:])
inp.requires_grad_()

in_feat = 8
out_feat = 4
batch_size = 3
momentum = 0

fc = nn.Linear(in_feat, out_feat)
fc.weight.data = cd.weights
fc.bias.data.zero_()

print(f"weights: {to_cpp(fc.weight.data)}")

optimizer = optim.SGD(fc.parameters(), lr=0.1, momentum=momentum)
loss_fn = nn.CrossEntropyLoss()

output = fc(inp)
output.requires_grad_()
output.retain_grad()

loss = loss_fn(output, cd.target_nonbinary)
loss.retain_grad()

loss.backward()
optimizer.step()

print(f"new_weights: {to_cpp(fc.weight.data)}")