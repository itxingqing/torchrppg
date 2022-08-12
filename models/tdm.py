import torch
import torch.nn as nn


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


# m = TDM(3)
# input = torch.randn(16, 256, 64, 24, 24)
# output = m(input)
# print(output.size())




