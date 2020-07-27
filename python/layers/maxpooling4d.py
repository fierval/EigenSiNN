import tstcommon.commondata4d as cd
import torch
import torch.nn as nn

pl = nn.MaxPool2d(2, stride=2, return_indices=True)
output, idx = pl(cd.inp)

fakeloss = torch.tensor([[[[0.53762484, 0.08841801],
          [0.26598877, 0.08771902]],

         [[0.14441812, 0.17294925],
          [0.08538872, 0.09313697]],

         [[0.27319407, 0.67958736],
          [0.58928680, 0.76245397]]],


        [[[0.93036085, 0.36994016],
          [0.41254914, 0.93899775]],

         [[0.51614463, 0.66111606],
          [0.01932418, 0.13548207]],

         [[0.78922635, 0.48408800],
          [0.61365259, 0.99544948]]]], dtype=torch.float, device="cpu")

output.backward(fakeloss)
