
import commondata4d as cd
from utils import to_cpp
from im2col import im2col_indices, col2im_indices

import torch
import torch.nn as nn

inp = cd.output
out_channels = 3
kernel_size = 3
padding = 1
batch_size = inp.shape[0]

# Same convolution
conv = nn.ConvTranspose2d(cd.inp.shape[1], out_channels, kernel_size,
padding=padding, dilation=2, bias=True)

conv.weight.data = cd.convweights
conv.bias.data = torch.zeros(out_channels)

output = conv(inp)
output.backward(cd.convlossTrans)

print(f"output {to_cpp(output)}")
print(f"dinput {to_cpp(inp.grad)}")
print (f"weights {to_cpp(conv.weight)}")
print(f"dweights {to_cpp(conv.weight.grad)}")


# now with im2col

n_filter = cd.convweights.shape[0]
x_col = im2col_indices(inp.detach().numpy(), kernel_size, kernel_size, padding=padding)
W_reshape = cd.convweights.detach().numpy().reshape(n_filter, -1)

dout_reshaped = cd.convloss.detach().numpy().transpose(1, 2, 3, 0).reshape(n_filter, -1)
if(padding == 1):
  dout_reshaped = cd.convlossPad1.detach().numpy().transpose(1, 2, 3, 0).reshape(n_filter, -1)

dX_col = W_reshape.T @ dout_reshaped

dX = col2im_indices(dX_col, cd.inp.shape, kernel_size, kernel_size, padding=padding)
