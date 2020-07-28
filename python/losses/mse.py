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
loss_fn = nn.MSELoss()

output = fc(inp)
output.requires_grad_()
output.retain_grad()

loss = loss_fn(output, cd.target)
loss.retain_grad()

loss.backward()
optimizer.step()

##################################################

loss_act = ((output - cd.target) ** 2).mean()

dav = 1./ (batch_size * out_feat) *torch.from_numpy(np.ones((batch_size, out_feat)))

dsq = 2 * dav * (output - cd.target)

print(f"output: {to_cpp(output)}")
print(f"dsq: {to_cpp(dsq)}")
print(f"y_grad: {to_cpp(output.grad)}")
print(f"loss: {loss}")