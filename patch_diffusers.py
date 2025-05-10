# patch_diffusers.py

import sys
import types

# Ces blocs ne sont pas utilisés par le pipeline officiel mais peuvent être nécessaires si on reconstruit UNet
from diffusers_helper.custom_blocks import HunyuanVideoDownBlock3D, HunyuanVideoUpBlock3D
from diffusers_helper.models.hunyuan_video_packed import AutoencoderKLHunyuanVideo

# Injecte dans diffusers.models.unet_3d_blocks
unet_3d_module = types.ModuleType("diffusers.models.unet_3d_blocks")
unet_3d_module.HunyuanVideoDownBlock3D = HunyuanVideoDownBlock3D
unet_3d_module.HunyuanVideoUpBlock3D = HunyuanVideoUpBlock3D
sys.modules["diffusers.models.unet_3d_blocks"] = unet_3d_module

# Injecte dans diffusers.models.autoencoder_kl
autoencoder_module = types.ModuleType("diffusers.models.autoencoder_kl")
autoencoder_module.AutoencoderKLHunyuanVideo = AutoencoderKLHunyuanVideo
sys.modules["diffusers.models.autoencoder_kl"] = autoencoder_module

print("🧩 Patches diffusers injectés avec succès.")

