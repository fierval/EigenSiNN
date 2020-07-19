import commondata4d as cd
from utils import to_cpp
from im2col import im2col_indices

import torch
import torch.nn as nn

inp = cd.inp.clone()
out_channels = 5
kernel_size = 3
padding = 0

col = im2col_indices(inp.detach().numpy(), kernel_size, kernel_size, padding=0)

# Same convolution
conv = nn.Conv2d(cd.inp.shape[1], out_channels, kernel_size,
 padding=padding, bias=False)

conv.weight.data = cd.convweights

output = conv(cd.inp)

output.backward(cd.convloss)

print(f"output {to_cpp(output)}")
print(f"dinput {to_cpp(cd.inp.grad)}")
print (f"weights {to_cpp(conv.weight)}")
print(f"dweights {to_cpp(conv.weight.grad)}")


