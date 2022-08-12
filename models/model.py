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
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


# model = Model()
# x = torch.randn((4, 256, 3, 36, 36))
# y = model(x)
# print(y.size())
