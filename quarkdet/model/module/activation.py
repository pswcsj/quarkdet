import torch.nn as nn

activations = {'ReLU': nn.ReLU,
               'LeakyReLU': nn.LeakyReLU,
               'ReLU6': nn.ReLU6,
               'SELU': nn.SELU,
               'ELU': nn.ELU,
               None: nn.Identity
               }


def act_layers(name):
    assert name in activations.keys()
    if name == 'LeakyReLU':
<<<<<<< HEAD
        return nn.LeakyReLU(negative_slope=0.1)
=======
        return nn.LeakyReLU(negative_slope=0.1, inplace=True)
>>>>>>> parent of b3ecc5b (Fix inplace = True to False in all ReLU and LeackyReLU)
    else:
        return activations[name](inplace=True)
