# custom_blocks.py (nettoyé – conservé si tu veux expérimenter, mais inutilisé ici)

import torch
import torch.nn as nn

class HunyuanVideoDownBlock3D(nn.Module):
    def __init__(self, in_channels=3, out_channels=8):
        super().__init__()
        self.resnets = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(1, out_channels),
            nn.SiLU()
        )
        self.downsamplers = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(1, out_channels),
            nn.SiLU()
        )

    def forward(self, x):
        B, C, T, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(-1, C, H, W)
        x = self.resnets(x)
        x = self.downsamplers(x)
        C_out, H_out, W_out = x.shape[1:]
        x = x.reshape(B, T, C_out, H_out, W_out).permute(0, 2, 1, 3, 4)
        return x

class HunyuanVideoUpBlock3D(nn.Module):
    def __init__(self, in_channels=8, out_channels=3):
        super().__init__()
        self.resnets = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.GroupNorm(1, in_channels),
            nn.SiLU()
        )
        self.upsamplers = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, C, T, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(-1, C, H, W)
        x = self.resnets(x)
        x = self.upsamplers(x)
        C_out, H_out, W_out = x.shape[1:]
        x = x.reshape(B, T, C_out, H_out, W_out).permute(0, 2, 1, 3, 4)
        return x
