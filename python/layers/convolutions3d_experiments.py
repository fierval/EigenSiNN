import numpy as np
import commondata4d as cd
from utils import to_cpp, is_eq
from im2col import im2col_indices, col2im_indices

import torch
import torch.nn as nn

inp = cd.output

out_channels = cd.convweights.shape[1]
in_channels = cd.convweights.shape[0]

kernel_size = 3
padding = 1
batch_size = inp.shape[0]
dilation = 2

# Same convolution
conv = nn.ConvTranspose2d(in_channels, cd.convweights.shape[0], kernel_size,
padding=padding, dilation=dilation, bias=True)

conv.weight.data = cd.convweights
conv.bias.data = torch.zeros(out_channels)

output = conv(inp)
output.backward(cd.convlossTrans)

# print(f"output {to_cpp(output)}")
# print(f"dinput {to_cpp(inp.grad)}")
# print (f"weights {to_cpp(conv.weight)}")
# print(f"dweights {to_cpp(conv.weight.grad)}")

######################## Transposed Kernel ##############################

conv_t = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation, bias=True)

inp.grad.zero_()

kernel = cd.convweights.permute(1, 0, 2, 3)
conv_t.weight.data = kernel
conv.bias.data = torch.zeros(out_channels)

output_t = conv_t.forward(inp)