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

    def forward(self, x):
        T, C, H, W = x.size()
        # x:(T, C, H, W) -> (H, W, C, T)
        x = torch.permute(x, (2, 3, 1, 0))
        # x:(H, W, C, T) -> (HxW, C, T)
        x = torch.reshape(x, (H*W, C, T))
        output = self.conv1d(x)
        output = torch.reshape(output, (H, W, C, T))
        # x:(H, W, C, T)->(T, C, H, W)
        output = torch.permute(output, (3, 2, 0, 1))
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
            # x:(T, C, H, W)->(T, C*N, H, W)
            output = torch.cat((output, x), dim=1)
        return output


# m = TDM(3)
# input = torch.randn(256, 32, 32, 32)
# output = m(input)
# print(output.size())




