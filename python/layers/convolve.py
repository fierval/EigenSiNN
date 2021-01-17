import numpy as np
import os

import torch
import torch.nn as nn

import commondata4d as cd
from utils import to_cpp

out_channels = 5
kernel_size = 3
padding = 1
batch_size = cd.inp.shape[0]
dilation = 2

# Same convolution
conv = nn.Conv2d(cd.inp.shape[1], out_channels, kernel_size, padding=padding, bias=False, dilation=dilation)

conv.weight.data = cd.convweights

output = conv(cd.inp)

output.backward(cd.convloss)

print(f"output {to_cpp(output)}")
print(f"dinput {to_cpp(cd.inp.grad)}")
print (f"weights {to_cpp(conv.weight)}")
print(f"dweights {to_cpp(conv.weight.grad)}")
