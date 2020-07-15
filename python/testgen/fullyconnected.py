import commondata2d as cd
import torch
import torch.nn as nn
from utils import to_cpp

in_feat = 8
out_feat = 4

fc = nn.Linear(in_feat, out_feat, bias=False)

fc.weight.data = torch.tensor([[ 0.30841491,  0.16301581,  0.05912393,  0.32572004, -0.00591815,
         -0.07333553,  0.16375038, -0.35274175],
        [ 0.19089887, -0.24521475,  0.27066174, -0.00526837, -0.18401390,
         -0.20650741, -0.28048125,  0.29642352],
        [-0.15496132,  0.15089461,  0.16939566, -0.25025240, -0.18078347,
         -0.07853529, -0.32877934,  0.19627282],
        [-0.28125578, -0.15781732, -0.32488498, -0.08520141, -0.27685770,
         -0.02988693,  0.18739149,  0.32216403]], requires_grad=True)

output = fc(cd.inp.reshape((3,8)))

fakeloss = torch.tensor([[0.13770211, 0.28582627, 0.86899745, 0.27578735],
        [0.04713255, 0.51820499, 0.27709258, 0.74432141],
        [0.47782332, 0.82197350, 0.52797425, 0.03082085]], requires_grad=True)
        
output.backward(fakeloss)

fakeloss_cpp = to_cpp(fakeloss)
inp_cpp = to_cpp(cd.inp.reshape((3,8)))
inp_grad_cpp = to_cpp(cd.inp.grad)
output_cpp = to_cpp(output)

print(f"fakeloss: {fakeloss_cpp}")
print(f"input: {cd.inp.reshape((3,8))}")
print(f"dL/dX: {inp_grad_cpp}")
print(f"output: {output_cpp}")