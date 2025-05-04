import os
import torch
import einops
import numpy as np
import safetensors.torch as sf

from PIL import Image
from diffusers import AutoencoderKLHunyuanVideo
from transformers import LlamaModel, CLIPTextModel, LlamaTokenizerFast, AutoTokenizer, SiglipImageProcessor, SiglipVisionModel
from diffusers_helper.hunyuan import encode_prompt_conds, vae_decode, vae_encode, vae_decode_fake
from diffusers_helper.utils import save_bcthw_as_mp4, crop_or_pad_yield_mask, soft_append_bcthw, resize_and_center_crop, generate_timestamp
from diffusers_helper.models.hunyuan_video_packed import HunyuanVideoTransformer3DModelPacked
from diffusers_helper.pipelines.k_diffusion_hunyuan import sample_hunyuan
from diffusers_helper.memory import gpu, get_cuda_free_memory_gb, move_model_to_device_with_memory_preservation, offload_model_from_device_for_memory_preservation, fake_diffusers_current_device, DynamicSwapInstaller, unload_complete_models, load_model_as_complete
from diffusers_helper.clip_vision import hf_clip_vision_encode
from diffusers_helper.bucket_tools import find_nearest_bucket


class FramePackAnimator:
    def __init__(self):
        self.high_vram = get_cuda_free_memory_gb(gpu) > 60

        # Load models
        self.text_encoder = LlamaModel.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='text_encoder', torch_dtype=torch.float16).cpu()
        self.text_encoder_2 = CLIPTextModel.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='text_encoder_2', torch_dtype=torch.float16).cpu()
        self.tokenizer = LlamaTokenizerFast.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='tokenizer')
        self.tokenizer_2 = AutoTokenizer.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='tokenizer_2')
        self.vae = AutoencoderKLHunyuanVideo.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='vae', torch_dtype=torch.float16).cpu()

        self.feature_extractor = SiglipImageProcessor.from_pretrained("lllyasviel/flux_redux_bfl", subfolder='feature_extractor')
        self.image_encoder = SiglipVisionModel.from_pretrained("lllyasviel/flux_redux_bfl", subfolder='image_encoder', torch_dtype=torch.float16).cpu()

        self.transformer = HunyuanVideoTransformer3DModelPacked.from_pretrained('lllyasviel/FramePackI2V_HY', torch_dtype=torch.bfloat16).cpu()

        self._prepare_models()

    def _prepare_models(self):
        for model in [self.vae, self.text_encoder, self.text_encoder_2, self.image_encoder, self.transformer]:
            model.eval()
            model.requires_grad_(False)

        if not self.high_vram:
            self.vae.enable_slicing()
            self.vae.enable_tiling()
            DynamicSwapInstaller.install_model(self.transformer, device=gpu)
            DynamicSwapInstaller.install_model(self.text_encoder, device=gpu)
        else:
            self.text_encoder.to(gpu)
            self.text_encoder_2.to(gpu)
            self.image_encoder.to(gpu)
            self.vae.to(gpu)
            self.transformer.to(gpu)

    @torch.no_grad()
    def animate(self, image: Image.Image, prompt: str, frames: int = 60, fps: int = 24, seed: int = 42) -> str:
        # Prepare input
        input_image = np.array(image)
        H, W, C = input_image.shape
        height, width = find_nearest_bucket(H, W, resolution=640)
        input_image_np = resize_and_center_crop(input_image, target_width=width, target_height=height)
        input_image_pt = torch.from_numpy(input_image_np).float() / 127.5 - 1
        input_image_pt = input_image_pt.permute(2, 0, 1)[None, :, None]

        # Encode
        llama_vec, clip_pooler = encode_prompt_conds(prompt, self.text_encoder, self.text_encoder_2, self.tokenizer, self.tokenizer_2)
        llama_vec_n, clip_pooler_n = torch.zeros_like(llama_vec), torch.zeros_like(clip_pooler)
        llama_vec, llama_mask = crop_or_pad_yield_mask(llama_vec, length=512)
        llama_vec_n, llama_mask_n = crop_or_pad_yield_mask(llama_vec_n, length=512)

        latent = vae_encode(input_image_pt, self.vae)
        image_encoder_output = hf_clip_vision_encode(input_image_np, self.feature_extractor, self.image_encoder)
        image_latents = image_encoder_output.last_hidden_state

        # Generate video
        latent_frames = sample_hunyuan(
            transformer=self.transformer,
            sampler='unipc',
            width=width,
            height=height,
            frames=frames,
            real_guidance_scale=1.0,
            distilled_guidance_scale=10.0,
            guidance_rescale=0.0,
            num_inference_steps=25,
            generator=torch.Generator("cpu").manual_seed(seed),
            prompt_embeds=llama_vec,
            prompt_embeds_mask=llama_mask,
            prompt_poolers=clip_pooler,
            negative_prompt_embeds=llama_vec_n,
            negative_prompt_embeds_mask=llama_mask_n,
            negative_prompt_poolers=clip_pooler_n,
            device=gpu,
            dtype=torch.bfloat16,
            image_embeddings=image_latents,
            latent_indices=torch.arange(frames).unsqueeze(0),
            clean_latents=latent,
            clean_latent_indices=torch.zeros((1, 1), dtype=torch.long),
            clean_latents_2x=latent,
            clean_latent_2x_indices=torch.zeros((1, 1), dtype=torch.long),
            clean_latents_4x=latent,
            clean_latent_4x_indices=torch.zeros((1, 1), dtype=torch.long),
        )

        # Decode
        video_tensor = vae_decode(latent_frames, self.vae).cpu()
        video_path = f"animation_{generate_timestamp()}.mp4"
        save_bcthw_as_mp4(video_tensor, video_path, fps=fps, crf=16)
        return video_path
