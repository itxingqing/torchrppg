import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import os


# -------------------------------------------------------------------------------------------------------------------
# PhysNet network
# -------------------------------------------------------------------------------------------------------------------
class PhysNetUpsample(nn.Module):
    def __init__(self, video_channels=3, model_channels=64):
        super().__init__()

        self.start = nn.Sequential(
            nn.Conv3d(in_channels=video_channels, out_channels=32, kernel_size=(1, 5, 5), stride=1, padding=(0, 2, 2)),
            nn.BatchNorm3d(32),
            nn.ELU()
        )

        # 1x
        self.loop1 = nn.Sequential(
            nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=0),
            nn.Conv3d(in_channels=32, out_channels=model_channels, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1)),
            nn.BatchNorm3d(model_channels),
            nn.ELU(),
            nn.Conv3d(in_channels=model_channels, out_channels=model_channels, kernel_size=(3, 3, 3), stride=1,
                      padding=(1, 1, 1)),
            nn.BatchNorm3d(model_channels),
            nn.ELU()
        )

        # encoder
        self.encoder1 = nn.Sequential(
            nn.AvgPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=0),
            nn.Conv3d(in_channels=model_channels, out_channels=model_channels, kernel_size=(3, 3, 3), stride=1,
                      padding=(1, 1, 1)),
            nn.BatchNorm3d(model_channels),
            nn.ELU(),
            nn.Conv3d(in_channels=model_channels, out_channels=model_channels, kernel_size=(3, 3, 3), stride=1,
                      padding=(1, 1, 1)),
            nn.BatchNorm3d(model_channels),
            nn.ELU(),
        )
        self.encoder2 = nn.Sequential(
            nn.AvgPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=0),
            nn.Conv3d(in_channels=model_channels, out_channels=model_channels, kernel_size=(3, 3, 3), stride=1,
                      padding=(1, 1, 1)),
            nn.BatchNorm3d(model_channels),
            nn.ELU(),
            nn.Conv3d(in_channels=model_channels, out_channels=model_channels, kernel_size=(3, 3, 3), stride=1,
                      padding=(1, 1, 1)),
            nn.BatchNorm3d(model_channels),
            nn.ELU()
        )

        #
        self.loop4 = nn.Sequential(
            nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=0),
            nn.Conv3d(in_channels=model_channels, out_channels=model_channels, kernel_size=(3, 3, 3), stride=1,
                      padding=(1, 1, 1)),
            nn.BatchNorm3d(model_channels),
            nn.ELU(),
            nn.Conv3d(in_channels=model_channels, out_channels=model_channels, kernel_size=(3, 3, 3), stride=1,
                      padding=(1, 1, 1)),
            nn.BatchNorm3d(model_channels),
            nn.ELU()
        )

        # decoder to reach back initial temporal length
        self.decoder1 = nn.Sequential(
            nn.Conv3d(in_channels=model_channels, out_channels=model_channels, kernel_size=(3, 1, 1), stride=1,
                      padding=(1, 0, 0)),
            nn.BatchNorm3d(model_channels),
            nn.ELU(),
        )
        self.decoder2 = nn.Sequential(
            nn.Conv3d(in_channels=model_channels, out_channels=model_channels, kernel_size=(3, 1, 1), stride=1,
                      padding=(1, 0, 0)),
            nn.BatchNorm3d(model_channels),
            nn.ELU()
        )

        self.end = nn.Sequential(
            nn.AdaptiveAvgPool3d((None, 1, 1)),
            nn.Conv3d(in_channels=model_channels, out_channels=1, kernel_size=(1, 1, 1), stride=1, padding=(0, 0, 0))
        )

        self.end_deploy = nn.Sequential(
            nn.AvgPool3d(kernel_size=(1, 8, 8), stride=(1, 1, 1), padding=0),
            nn.Conv3d(in_channels=model_channels, out_channels=1, kernel_size=(1, 1, 1), stride=1, padding=(0, 0, 0))
        )

    def forward(self, x):
        B, C, T, H, W = x.size()
        # x_mean = torch.mean(x, dim=(2, 3, 4), keepdim=True)
        # x_std = torch.std(x, dim=(2, 3, 4), keepdim=True)
        # x = (x - x_mean) / x_std
        parity = []
        x = self.start(x)
        x = self.loop1(x)
        parity.append(x.size(2) % 2)
        x = self.encoder1(x)
        parity.append(x.size(2) % 2)
        x = self.encoder2(x)
        x = self.loop4(x)

        x = F.interpolate(x, scale_factor=(2, 1, 1))
        x = self.decoder1(x)
        x = F.pad(x, (0, 0, 0, 0, 0, parity[-1]), mode='replicate')
        x = F.interpolate(x, scale_factor=(2, 1, 1))
        x = self.decoder2(x)
        x = F.pad(x, (0, 0, 0, 0, 0, parity[-2]), mode='replicate')
        x = self.end(x)
        x = x.view(-1, T)

        return x

    def forward_deploy(self, x):
        # B, C, T, H, W = x.size()
        # x_mean = torch.mean(x, dim=(2, 3, 4), keepdim=True)
        # x_std = torch.std(x, dim=(2, 3, 4), keepdim=True)
        # x = (x - x_mean) / x_std
        # parity = []
        x = self.start(x)
        x = self.loop1(x)
        # parity.append(x.size(2) % 2)
        x = self.encoder1(x)
        # parity.append(x.size(2) % 2)
        x = self.encoder2(x)
        x = self.loop4(x)

        x = F.interpolate(x, scale_factor=(2, 1, 1))
        x = self.decoder1(x)
        # x = F.pad(x, (0, 0, 0, 0, 0, 0), mode='replicate')
        x = F.interpolate(x, scale_factor=(2, 1, 1))
        x = self.decoder2(x)
        # x = F.pad(x, (0, 0, 0, 0, 0, 0), mode='replicate')
        x = self.end_deploy(x)
        x = x.view(-1, 240)

        return x

if __name__ == '__main__':
    model = PhysNetUpsample()
    x = torch.randn((1, 3, 240, 36, 36))
    rPPG = model(x)
    print(rPPG.shape)