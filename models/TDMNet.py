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

class DTC(nn.Module):
    def __init__(self):
        super(DTC, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2)
        t1 = -2* torch.ones(32, 32, 1)
        t2 = -1*torch.ones(32, 32, 1)
        t3 = 0 * torch.ones(32, 32, 1)
        t4 = 1 * torch.ones(32, 32, 1)
        t5 = 2 * torch.ones(32, 32, 1)
        self.conv1d.weight.data = torch.cat((t1, t2, t3, t4, t5), dim=2)
        self.conv1d.weight.requires_grad = False
        self.bn = nn.BatchNorm2d(32)
        self.tanh = nn.Tanh()
        self.dp = nn.Dropout(p=0.5)

    def forward(self, x):
        B, T, C, H, W = x.size()
        # x:(B, T, C, H, W) -> (B, H, W, C, T)
        x = torch.permute(x, (0, 3, 4, 2, 1))
        # x:(B, H, W, C, T) -> (B*HxW, C, T)
        x = torch.reshape(x, (B*H*W, C, T))
        output = self.conv1d(x)
        # x:(B*HxW, C, T) -> (B, H, W, C, T)
        output = torch.reshape(output, (B, H, W, C, T))
        # x:(B, H, W, C, T) -> (B, T, C, H, W)
        output = torch.permute(output, (0, 4, 3, 1, 2))
        # x:(B, T, C, H, W)-> (B*T, C, H, W)
        output = torch.reshape(output, (B*T, C, H, W))
        output = self.tanh(output)
        output = self.bn(output)
        output = self.dp(output)
        # x:(B*T, C, H, W)-> (B, T, C, H, W)
        output = torch.reshape(output, (B, T, C, H, W))
        return output


class TDM(nn.Module):
    def __init__(self, N):
        super(TDM, self).__init__()
        self.tdm = DTC()
        self.number = N

    def forward(self, x):
        x = self.tdm(x)
        output = x
        for i in range(1, self.number):
            x = self.tdm(x)
            # x:(B, T, C, H, W)->(B, T, C*N, H, W)
            output = torch.cat((output, x), dim=2)
        return output


class TDMNet(nn.Sequential):
    def __init__(self):
        super(TDMNet, self).__init__(
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


# m = TDM(3)
# input = torch.randn(16, 256, 64, 24, 24)
# output = m(input)
# print(output.size())




