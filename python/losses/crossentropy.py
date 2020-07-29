import tstcommon.commondata2d as cd

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

fc = nn.Linear(in_feat, out_feat, bias=False)

fc.weight.data = cd.weights
optimizer = optim.SGD(fc.parameters(), lr=0.01, momentum=0.9)
loss_fn = nn.CrossEntropyLoss()

output = fc(inp)
output.requires_grad_()
output.retain_grad()

loss = loss_fn(output, cd.target_nonbinary)
loss.retain_grad()

loss.backward()
optimizer.step()

##################################################

def cross_entropy(o, target):
  loss_actual = []
  for i in range(o.shape[0]):
    loss_actual.append(-o[i, target[i]] + torch.log(torch.exp(o[i, :]).sum()))

  return torch.tensor(loss_actual).mean()

cross_entropy(output, cd.target_nonbinary)

def cross_entropy_grad(o):
  mask = cd.target

  N, C = o.shape

  dxi = -1. / N * mask
  dsum = 1./N * torch.from_numpy(np.ones((N)))
  dlog = 1. / torch.exp(o).sum(axis=1) * dsum
  dsumexp = torch.from_numpy(np.ones((N, C))) * torch.exp(o) * dlog.reshape((N, 1))

  dL = dxi + dsumexp
  return dL

cross_entropy_grad(output)

