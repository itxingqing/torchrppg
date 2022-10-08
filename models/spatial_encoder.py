import torch
import torch.nn as nn


class ConvTanhBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=1):
        super(ConvTanhBN, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False),
            nn.Tanh(),
            nn.BatchNorm2d(out_channels, momentum=0.1),
        )


class ConvBNTanh(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=1):
        super(ConvBNTanh, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels, momentum=0.1),
            nn.ReLU(),
        )


class SpatialEncoder(nn.Module):
    def __init__(self):
        super(SpatialEncoder, self).__init__()
        self.conv1 = ConvTanhBN(in_channels=3, out_channels=16)
        self.pool1 = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv2 = ConvTanhBN(in_channels=16, out_channels=32)
        self.pool2 = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))

    def forward(self, x):
        # x_mean = torch.mean(x, dim=(1, 3, 4), keepdim=True)
        # x_std = torch.std(x, dim=(1, 3, 4), keepdim=True)
        # x = (x - x_mean) / x_std
        B, T, C, H, W = x.size()
        # (B, T, 3, 96, 96) -> (B, T, 16, 96, 96)
        x = torch.reshape(x, (B*T, C, H, W))
        x = self.conv1(x)
        # (B*T, 16, 96, 96) -> (B*T, 16, 48, 48)
        x = self.pool1(x)
        # (B*T, 16, 48, 48) -> (B*T, 32, 48, 48)
        x = self.conv2(x)
        # (B*T, 32, 48, 48) -> (B*T, 32, 24, 24)
        x = self.pool2(x)
        _, C, H, W = x.size()
        x = torch.reshape(x, (B, T, C, H, W))
        return x


# model = SpatialEncoder()
# x = torch.randn((16, 256, 3, 96, 96))
# x.cuda()
# y = model(x).cuda()
# print(y.size())