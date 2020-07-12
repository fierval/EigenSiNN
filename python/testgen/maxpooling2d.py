import commondata2d as cd
import torch
import torch.nn as nn

inp = torch.randn(1, 3, 8)
pl = nn.MaxPool1d(4, stride=2, return_indices=True)
output, idx = pl(inp)

fakeloss = torch.tensor([[[0.31773561, 0.25510252, 0.73881042],
         [0.81441122, 0.74392009, 0.56959468],
         [0.94542354, 0.31825888, 0.96742082]]])

output.backward(fakeloss)
