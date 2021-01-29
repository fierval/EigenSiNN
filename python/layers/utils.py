import re
import torch
import numpy as np

def to_cpp(x):

    s = str(x).replace("[", "{").replace("]", "}").replace("tensor", "")

    # remove any other tensor parameters at the end
    regex = r"\,\s+[A-z]+.+\)$"
    s = re.sub(regex, ')', s)

    return s

def is_eq(x, y, prec=1e-5):

  x_ = x.cpu().detach().numpy() if torch.is_tensor(x) else x
  y_ = y.cpu().detach().numpy() if torch.is_tensor(y) else y

  return np.all((x_-y_) <= prec)
