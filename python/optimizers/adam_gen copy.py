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
epochs = 1
betas = (0.9, 0.999)
eps = 1e-08

fc = nn.Linear(in_feat, out_feat)
fc.weight.data = cd.weights
fc.bias.data.zero_()

print(f"weights: {to_cpp(fc.weight.data)}")

optimizer = optim.Adam(fc.parameters(), lr=0.001, betas=betas, eps=eps)
loss_fn = nn.CrossEntropyLoss()

for i in range(epochs):

  fc.zero_grad()
  optimizer.zero_grad()
  
  output = fc(inp)
  output.requires_grad_()

  loss = loss_fn(output, cd.target_nonbinary)
  
  loss.backward()
  optimizer.step()

print(f"new_weights: {to_cpp(fc.weight.data)}")
print(f"new_bias: {to_cpp(fc.bias.data)}")