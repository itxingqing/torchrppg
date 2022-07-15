import torch
import torch.nn as nn
from .spatial_encoder import SpatialEncoder
from .tdm import TDM
from .head import Head


class Model(nn.Sequential):
    def __init__(self):
        super(Model, self).__init__(
            SpatialEncoder(),
            TDM(3),
            Head()
        )


# model = Model()
# x = torch.randn((4, 256, 3, 36, 36))
# y = model(x)
# print(y.size())
