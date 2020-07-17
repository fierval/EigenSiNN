import commondata4d as cd
from utils import to_cpp

import torch
import torch.nn as nn

inp = cd.inp
out_channels = 5
kernel_size = 3
padding = 1

# Same convolution
conv = nn.Conv2d(cd.inp.shape[1], out_channels, kernel_size,
 padding=padding, bias=False)

conv.weight.data = cd.convweights

output = conv(inp)

output.backward(cd.convloss)

print(f"output {to_cpp(output)}")
print(f"dinput {to_cpp(inp.grad)}")
print (f"weights {to_cpp(conv.weight)}")
print(f"dweights {to_cpp(conv.weight.grad)}")
