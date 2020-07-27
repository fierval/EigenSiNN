
import tstcommon.commondata4d as cd
from utils import to_cpp
from im2col import im2col_indices, col2im_indices

import torch
import torch.nn as nn

inp = cd.inp.clone()
out_channels = 5
kernel_size = 3
padding = 0
batch_size = inp.shape[0]

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

# now with im2col

n_filter = cd.convweights.shape[0]
x_col = im2col_indices(inp.detach().numpy(), kernel_size, kernel_size, padding=padding)
W_reshape = cd.convweights.detach().numpy().reshape(n_filter, -1)

dout_reshaped = cd.convloss.detach().numpy().transpose(1, 2, 3, 0).reshape(n_filter, -1)
dX_col = W_reshape.T @ dout_reshaped

dX = col2im_indices(dX_col, inp.shape, kernel_size, kernel_size, padding=padding)
