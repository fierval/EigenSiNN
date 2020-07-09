import commondata
import torch
import torch.nn as nn

bn = nn.BatchNorm2d(3, eps=1e-5, momentum=0.1)
bn.weight.data = \
    torch.tensor([1., 2., 3.,], dtype=torch.float, device=commondata.device)

bn.bias.data = \
    torch.tensor([0.1, 0.2, 0.3,], dtype=torch.float, device=commondata.device)


output = bn(commondata.inp)
output.backward(commondata.fakeloss)

