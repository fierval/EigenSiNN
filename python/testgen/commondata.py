import torch
import torch.nn as nn

device = torch.device("cpu")
torch.set_printoptions(precision=8)

inp = torch.tensor([[[[0.04538226, 0.95949411, 0.50452268, 0.59602135],
          [0.74790466, 0.73222357, 0.07335120, 0.85607505],
          [0.90876633, 0.47080678, 0.24576986, 0.59051216],
          [0.29745585, 0.98773926, 0.62964255, 0.36631453]],

         [[0.50406832, 0.42415136, 0.34337270, 0.72482306],
          [0.46068907, 0.96129739, 0.62765759, 0.27029365],
          [0.35458261, 0.38653296, 0.47089547, 0.82289129],
          [0.63725311, 0.29671627, 0.54370487, 0.87211448]],

         [[0.85018283, 0.75999165, 0.15591985, 0.40320259],
          [0.74964321, 0.10842186, 0.55331987, 0.58271384],
          [0.67342269, 0.74921608, 0.96604007, 0.26602185],
          [0.53013945, 0.98281318, 0.43664277, 0.76730776]]],


        [[[0.61419797, 0.84584051, 0.52509171, 0.18094194],
          [0.38709599, 0.25413889, 0.74516875, 0.27006137],
          [0.72085720, 0.84589291, 0.62312174, 0.93618810],
          [0.64560336, 0.61316019, 0.96158671, 0.92482752]],

         [[0.11359078, 0.84837216, 0.81256318, 0.98481309],
          [0.36437154, 0.10838908, 0.87681556, 0.37505645],
          [0.05423701, 0.66477776, 0.46080470, 0.09901488],
          [0.24049366, 0.78823119, 0.09643930, 0.77357787]],

         [[0.92674822, 0.49461865, 0.82311010, 0.50724411],
          [0.43091267, 0.64998370, 0.53645664, 0.69436163],
          [0.01849866, 0.22924691, 0.96253598, 0.99673647],
          [0.09387362, 0.86051500, 0.52929527, 0.41473496]]]], device=device, dtype=torch.float,
       requires_grad=True) 

fakeloss = torch.tensor([[[[0.03411579, 0.30640966, 0.19440770, 0.00902396],
          [0.87998140, 0.08524799, 0.20483696, 0.11298430],
          [0.33172631, 0.01118428, 0.81524986, 0.33781791],
          [0.72575861, 0.48591703, 0.83571565, 0.74452734]],

         [[0.99675554, 0.52758676, 0.70237589, 0.33300936],
          [0.18728876, 0.86973250, 0.55728424, 0.77503520],
          [0.90626729, 0.46376646, 0.63194448, 0.14518911],
          [0.89655298, 0.54174763, 0.61954391, 0.52469307]],

         [[0.73198414, 0.08069026, 0.49224281, 0.47842449],
          [0.98264766, 0.13089573, 0.07510883, 0.34220546],
          [0.08220178, 0.48254997, 0.63483059, 0.02323723],
          [0.77521580, 0.24670333, 0.50362766, 0.62766153]]],


        [[[0.98965490, 0.88055056, 0.92259568, 0.24774176],
          [0.72176152, 0.90675122, 0.08763289, 0.57801038],
          [0.17916220, 0.27972364, 0.25751638, 0.81748664],
          [0.56826556, 0.73269165, 0.31478405, 0.95569855]],

         [[0.51677078, 0.98813075, 0.33273274, 0.64959395],
          [0.08583117, 0.13452137, 0.38807088, 0.40816027],
          [0.35822779, 0.15018040, 0.96034634, 0.90043443],
          [0.91827869, 0.20131719, 0.65043378, 0.43243688]],

         [[0.91697454, 0.39756697, 0.30185652, 0.28277457],
          [0.81693399, 0.69717938, 0.09399003, 0.18187153],
          [0.63124579, 0.38466102, 0.76345491, 0.11254698],
          [0.19470620, 0.97134984, 0.39794821, 0.64935327]]]], device=device, dtype=torch.float)
