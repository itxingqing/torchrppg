import torch
import torch.nn as nn


class Head(nn.Module):
    def __init__(self):
        super(Head, self).__init__()
        self.head = nn.AdaptiveAvgPool2d(1)
        self.channel_wise_conv = nn.Conv2d(in_channels=96*240, out_channels=240, kernel_size=1)

    def forward(self, x):
        # B, T, CN, H, W = x.size()
        # # (B, T, CN , H, W) -> (B*T, CN , H, W)
        # x = torch.reshape(x, (B*T, CN, H, W))
        # # (B*T, CN , H, W) -> (B*T, CN, 1)
        # x = self.head(x)
        # # (B*T, CN , 1, 1) -> (B*T, 1, 1, 1)
        # x = self.channel_wise_conv(x)
        # # (B*T, 1, 1, 1) -> (B, T, 1)
        # x = torch.reshape(x, (B, T))

        B, T, CN, H, W = x.size()
        # (B, T, CN , H, W) -> (B, T*CN , H, W)
        x = torch.reshape(x, (B, T * CN, H, W))
        # (B, T*CN , H, W) -> (B, T*CN, 1, 1)
        x = self.head(x)
        # (B, T*CN , 1, 1) -> (B, T, 1, 1)
        x = self.channel_wise_conv(x)
        # (B, T, 1, 1) -> (B, T, 1)
        x = torch.reshape(x, (B, T))

        return x


# m = Head()
# input = torch.randn(16, 256, 192, 24, 24)
# output = m(input)
# print(output.size())

