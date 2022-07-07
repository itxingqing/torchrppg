import torch
import torch.nn as nn


class Head(nn.Module):
    def __init__(self):
        super(Head, self).__init__()
        self.head = nn.AdaptiveAvgPool2d(1)
        self.channel_wise_conv = nn.Conv2d(in_channels=96, out_channels=1, kernel_size=1)

    def forward(self, x):
        x = self.head(x)
        x = self.channel_wise_conv(x)
        x = torch.squeeze(x)
        return x


# m = Head()
# input = torch.randn(256, 96, 32, 32)
# output = m(input)
# print(output.size())

