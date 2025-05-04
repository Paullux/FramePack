from cog import BasePredictor, Path, Input
from PIL import Image
import torch
import os
import einops
import numpy as np
import shutil

from transformers import (
    CLIPTextModel, AutoTokenizer,
    SiglipImageProcessor, SiglipVisionModel,
    LlamaModel, LlamaTokenizerFast
)
from diffusers import AutoencoderKLHunyuanVideo
from diffusers_helper.models.hunyuan_video_packed import HunyuanVideoTransformer3DModelPacked
from diffusers_helper.hunyuan import encode_prompt_conds, vae_decode_fake
from diffusers_helper.utils import generate_timestamp, resize_and_center_crop, save_bcthw_as_mp4, crop_or_pad_yield_mask
from diffusers_helper.clip_vision import hf_clip_vision_encode
from diffusers_helper.pipelines.k_diffusion_hunyuan import sample_hunyuan

class Predictor(BasePredictor):
    def setup(self):
        print("🔧 Chargement des modèles...")
        self.vae = AutoencoderKLHunyuanVideo.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder="vae", torch_dtype=torch.float16).to("cuda")
        self.text_encoder = LlamaModel.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder="text_encoder", torch_dtype=torch.float16).to("cuda")
        self.text_encoder_2 = CLIPTextModel.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder="text_encoder_2", torch_dtype=torch.float16).to("cuda")
        self.tokenizer = LlamaTokenizerFast.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder="tokenizer")
        self.tokenizer_2 = AutoTokenizer.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder="tokenizer_2")
        self.image_encoder = SiglipVisionModel.from_pretrained("lllyasviel/flux_redux_bfl", subfolder="image_encoder", torch_dtype=torch.float16).to("cuda")
        self.processor = SiglipImageProcessor.from_pretrained("lllyasviel/flux_redux_bfl", subfolder="feature_extractor")
        self.transformer = HunyuanVideoTransformer3DModelPacked.from_pretrained("lllyasviel/FramePackI2V_HY", torch_dtype=torch.bfloat16).to("cuda")
        self.vae.eval()
        self.text_encoder.eval()
        self.image_encoder.eval()
        self.transformer.eval()

    def predict(
        self,
        image: Path = Input(description="Image (.png/.jpg)", default=None),
        prompt: str = Input(description="Description du mouvement", default="The character moves confidently."),
        seed: int = Input(description="Seed aléatoire", default=42),
        steps: int = Input(description="Nombre d'étapes de sampling", default=25),
        duration_seconds: float = Input(description="Durée de la vidéo en secondes", default=5.0),
        fps: int = Input(description="Framerate de la vidéo", default=30),
    ) -> Path:
        print("🚀 Génération en cours...")
        generator = torch.Generator(device="cuda").manual_seed(seed)

        # Préparation image
        img = Image.open(image).convert("RGB")
        np_img = np.array(img)
        height, width = 512, 512
        input_image_np = resize_and_center_crop(np_img, target_width=width, target_height=height)
        input_image_pt = torch.from_numpy(input_image_np).float() / 127.5 - 1
        input_image_pt = input_image_pt.permute(2, 0, 1)[None, :, None].to("cuda")

        # Encodage texte avec LLaMA + CLIP
        prompt_embeds, prompt_poolers = encode_prompt_conds(
            prompt, self.text_encoder, self.text_encoder_2, self.tokenizer, self.tokenizer_2
        )

        # Masques pour guidance
        prompt_embeds, prompt_mask = crop_or_pad_yield_mask(prompt_embeds, length=512)
        neg_prompt_embeds = torch.zeros_like(prompt_embeds)
        neg_prompt_mask = torch.zeros_like(prompt_mask)
        neg_prompt_poolers = torch.zeros_like(prompt_poolers)

        # Encodage VAE
        start_latent = self.vae.encode(input_image_pt).latent_dist.sample().to(dtype=self.vae.dtype)

        # Encodage image CLIP
        img_enc_out = hf_clip_vision_encode(input_image_np, self.processor, self.image_encoder)
        clip_hidden = img_enc_out.last_hidden_state.to(dtype=torch.bfloat16)

        # Sampling
        latent_frames = sample_hunyuan(
            transformer=self.transformer,
            sampler='unipc',
            width=width,
            height=height,
            frames=int(fps * duration_seconds),
            real_guidance_scale=1.0,
            distilled_guidance_scale=10.0,
            guidance_rescale=0.0,
            num_inference_steps=steps,
            generator = generator,
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_mask,
            prompt_poolers=prompt_poolers,
            negative_prompt_embeds=neg_prompt_embeds,
            negative_prompt_embeds_mask=neg_prompt_mask,
            negative_prompt_poolers=neg_prompt_poolers,
            device="cuda",
            dtype=torch.bfloat16,
            image_embeddings=clip_hidden,
            latent_indices=None,
            clean_latents=start_latent,
            clean_latent_indices=None,
            clean_latents_2x=None,
            clean_latent_2x_indices=None,
            clean_latents_4x=None,
            clean_latent_4x_indices=None,
        )

        # Decode + Save
        pixel_frames = vae_decode_fake(latent_frames)
        video_np = (pixel_frames * 255.0).detach().cpu().numpy().clip(0, 255).astype(np.uint8)
        video_np = einops.rearrange(video_np, 'b c t h w -> t h w c')

        out_file = f"/tmp/{generate_timestamp()}.mp4"
        save_bcthw_as_mp4(video_np, out_file, fps=fps)

        return Path(out_file)
