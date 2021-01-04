import numpy as np
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
dilation = 2

# Same convolution
conv = nn.ConvTranspose2d(cd.inp.shape[1], out_channels, kernel_size,
padding=padding, dilation=dilation, bias=True)

conv.weight.data = cd.convweights
conv.bias.data = torch.zeros(out_channels)

output = conv(inp)
output.backward(cd.convlossTrans)

print(f"output {to_cpp(output)}")
print(f"dinput {to_cpp(inp.grad)}")
print (f"weights {to_cpp(conv.weight)}")
print(f"dweights {to_cpp(conv.weight.grad)}")

########################## now with im2col ####################################################

# dilate the kernel
dilated_size = dilation * (kernel_size - 1) + 1
dilated = np.zeros((cd.convweights.shape[0], out_channels, dilated_size, dilated_size))

kernel = conv.weight.data.detach().numpy()

w_dilated = [dilation * w for w in range(kernel_size)]
h_dilated = [dilation * h for h in range(kernel_size)]
dilated[:, :, h_dilated, w_dilated] = kernel[:, :, range(kernel_size), range(kernel_size)]

# forward pass with transposed kernel
n_filter = cd.convweights.shape[0]
inp_reshaped = inp.detach().numpy().transpose(1, 2, 3, 0).reshape(n_filter, -1)
W_reshape = dilated.reshape(n_filter, -1)

out_col = W_reshape.T @ inp_reshaped
# image is recovered through col2im
out_image = col2im_indices(out_col, cd.inp.shape, dilated_size, dilated_size, padding=padding)

# x_col = im2col_indices(inp.detach().numpy(), kernel_size, dilated_size, padding=padding)
# W_reshape = cd.convweights.detach().numpy().reshape(n_filter, -1)

# dout_reshaped = cd.convloss.detach().numpy().transpose(1, 2, 3, 0).reshape(n_filter, -1)
# if(padding == 1):
#   dout_reshaped = cd.convlossTrans.detach().numpy().transpose(1, 2, 3, 0).reshape(n_filter, -1)

# dX_col = W_reshape.T @ dout_reshaped

# dX = col2im_indices(dX_col, cd.inp.shape, kernel_size, kernel_size, padding=padding)
