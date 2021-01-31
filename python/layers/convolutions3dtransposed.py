import numpy as np
import commondata4d as cd
from utils import to_cpp, is_eq
from im2col import im2col_indices, col2im_indices

import torch
import torch.nn as nn

inp = cd.output
# out and in channels are transposed vs ordinary convolution
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

print(f"output {to_cpp(output)}")
print(f"dinput {to_cpp(inp.grad)}")
print (f"weights {to_cpp(conv.weight)}")
print(f"dweights {to_cpp(conv.weight.grad)}")

########################## now with im2col ####################################################

### Forward Pass
# dilate the kernel
dilated_size = dilation * (kernel_size - 1) + 1
dilated = np.zeros((cd.convweights.shape[0], out_channels, dilated_size, dilated_size))

kernel = conv.weight.data.detach().numpy()

for h in range(kernel_size):
  for w in range(kernel_size):
    dilated[:, :, dilation * h, dilation * w] = kernel[:, :, h, w]

# forward pass with transposed kernel
n_filters = cd.convweights.shape[0]
inp_reshaped = inp.detach().numpy().transpose(1, 2, 3, 0).reshape(n_filters, -1)
W_reshape = dilated.reshape(n_filters, -1)

out_col = W_reshape.T @ inp_reshaped

# image is recovered through col2im
out_image = col2im_indices(out_col, cd.inp.shape, dilated_size, dilated_size, padding=padding)
is_eq(output, out_image)

################################################################################################
### Backward Pass
dout_col = im2col_indices(cd.convlossTrans.detach().numpy(), dilated_size, dilated_size, padding=padding)
#dout_reshaped = cd.convlossTrans.detach().numpy().transpose(1, 2, 3, 0).reshape(n_filter, -1)
dX_col = W_reshape @ dout_col
c, _, h, w = inp.shape
dX = dX_col.reshape(n_filters, h, w, c).transpose(3, 0, 1, 2)
is_eq(dX, inp.grad)

dW = dout_col @ inp_reshaped.T
dW = dW.reshape(dilated.shape)