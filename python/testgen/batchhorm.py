import torch
import torch.nn as nn

device = torch.device("cpu")
torch.set_printoptions(precision=8)

inp = torch.tensor([[0.56279999, 0.93430001, 0.25929999, 0.79210001, 0.15889999],
        [0.18510000, 0.04310000, 0.50970000, 0.88209999, 0.58310002]], device=device, dtype=torch.float,
       requires_grad=True) 

bn = nn.BatchNorm1d(5, eps=1e-5, momentum=0.1)
bn.weight.data = \
    torch.tensor([1., 2., 3., 4., 5.], dtype=torch.float, device=device)

bn.bias.data = \
    torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5], dtype=torch.float, device=device)

o = torch.tensor([[0.07951424, 0.39795890, 0.48816258, 0.58650136, 0.80818069],
        [0.33679566, 0.74452204, 0.24355969, 0.36228219, 0.69534987]], device=device, dtype=torch.float)

output = bn(inp)
output.backward(o)

