import torch.nn as nn
from diffusers.models.modeling_utils import ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config

from diffusers_helper.custom_blocks import HunyuanVideoDownBlock3D, HunyuanVideoUpBlock3D

class AutoencoderKLHunyuanVideo(ModelMixin, ConfigMixin, nn.Module):
    @register_to_config
    def __init__(self):
        super().__init__()
        self.encoder = HunyuanVideoDownBlock3D(in_channels=3, out_channels=8)
        self.decoder = HunyuanVideoUpBlock3D(in_channels=8, out_channels=3)
        self.quant_conv = nn.Identity()
        self.post_quant_conv = nn.Identity()

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)
