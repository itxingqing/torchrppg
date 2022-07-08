import torch
import torch.nn as nn


class ConvTanhBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=1):
        super(ConvTanhBN, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False),
            nn.Tanh(),
            nn.BatchNorm2d(out_channels, momentum=0.1),
        )


class SpatialEncoder(nn.Module):
    def __init__(self):
        super(SpatialEncoder, self).__init__()
        self.conv1 = ConvTanhBN(in_channels=3, out_channels=16)
        self.pool1 = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv2 = ConvTanhBN(in_channels=16, out_channels=32)
        self.pool2 = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))

    def forward(self, x):
        B, T, C, H, W = x.size()
        # (B, T, 3, 36, 36) -> (B, T, 16, 36, 36)
        x = torch.reshape(x, (B*T, C, H, W))
        x = self.conv1(x)
        # (B*T, 16, 36, 36) -> (B*T, 16, 18, 18)
        x = self.pool1(x)
        # (B*T, 16, 18, 18) -> (B*T, 32, 18, 18)
        x = self.conv2(x)
        # (B*T, 32, 18, 18) -> (B*T, 32, 9, 9)
        x = self.pool2(x)
        _, C, H, W = x.size()
        x = torch.reshape(x, (B, T, C, H, W))
        return x


# model = SpatialEncoder()
# x = torch.randn((16, 256, 3, 128, 128))
# y = model(x)
# print(y.size())