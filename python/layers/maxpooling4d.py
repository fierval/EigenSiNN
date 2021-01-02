import sys, os

dir = os.path.join(os.path.dirname(__file__), 'tstcommon')

sys.path.insert(0, dir)

import commondata4d as cd
from utils import to_cpp
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
cd.inp.grad.data.zero_()

pl1 = nn.MaxPool2d(kernel_size=2, stride=1, return_indices=True)
output1, idx1 = pl1(cd.inp)
fakeloss1 = torch.tensor([[[[0.18871814, 0.66104203, 0.67412460],
          [0.68415672, 0.83918720, 0.16296452],
          [0.57792860, 0.44864488, 0.01823634]],

         [[0.04834914, 0.10648906, 0.18182540],
          [0.43325162, 0.04500020, 0.48198962],
          [0.79064542, 0.13256913, 0.77111048]],

         [[0.87654692, 0.81638652, 0.65065503],
          [0.03452933, 0.64253539, 0.03129411],
          [0.71614343, 0.87064117, 0.46728039]]],


        [[[0.93335724, 0.77388012, 0.24067771],
          [0.24605733, 0.29329503, 0.17910063],
          [0.69973481, 0.71793348, 0.11202455]],

         [[0.39639467, 0.08519226, 0.50150609],
          [0.03796721, 0.32585174, 0.66033578],
          [0.68264580, 0.23440838, 0.35360551]],

         [[0.28318352, 0.95920134, 0.18036056],
          [0.10765439, 0.64352757, 0.95874316],
          [0.37381238, 0.44888192, 0.00114185]]]], dtype=torch.float, device="cpu")
output1.backward(fakeloss1)

print(to_cpp(cd.inp.grad))
cd.inp.grad.data.zero_()