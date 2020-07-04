import torch
import numpy as np
import torch.nn as nn

device = torch.device("cuda")


input = torch.tensor([[0.5628, 0.9343, 0.2593, 0.7921, 0.1589],
        [0.1851, 0.0431, 0.5097, 0.8821, 0.5831]], device='cuda:0',
       requires_grad=True)

bn = nn.BatchNorm1d(5, eps=1e-5, momentum=0.1).cuda()
o = torch.ones((2, 5), device=device, dtype=float)

output = bn(input)