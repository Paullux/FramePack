import os
import glob
import tempfile
from PIL import Image
import torch
from cog import BasePredictor, Input, Path
import accelerate

# Patch les modules `diffusers` avec nos classes personnalisées
import patch_diffusers

from diffusers import HunyuanVideoPipeline, AutoencoderKLHunyuanVideo
from diffusers_helper.pipelines.k_diffusion_hunyuan import sample_hunyuan
# from diffusers_helper.models.hunyuan_video_packed import AutoencoderKLHunyuanVideo

# Définit le cache local (non strictement nécessaire avec chemin direct)
os.environ["HF_HOME"] = "/src/models/hf_cache"

print(">>> 📁 Scan des fichiers présents (debug)")
for f in glob.glob("/src/hf_download/hub/models--hunyuanvideo-community--HunyuanVideo/**", recursive=True):
    if "config.json" in f or "safetensors" in f:
        print("   -", f)

class Predictor(BasePredictor):
    def setup(self):
        import os
        import torch
        from diffusers import AutoencoderKLHunyuanVideo
        from transformers import LlamaModel, CLIPTextModel, LlamaTokenizerFast, CLIPTokenizer, SiglipImageProcessor, SiglipVisionModel
        from diffusers_helper.models.hunyuan_video_packed import HunyuanVideoTransformer3DModelPacked

        model_path = "/src/hf_download/hub/models--hunyuanvideo-community--HunyuanVideo/snapshots/e8c2aaa66fe3742a32c11a6766aecbf07c56e773"

        print(f"🔧 Chargement des composants du modèle depuis {model_path}")

        # Load image_embeddings
        self.image_processor = SiglipImageProcessor.from_pretrained("lllyasviel/flux_redux_bfl", subfolder="feature_extractor")
        self.image_encoder = SiglipVisionModel.from_pretrained("lllyasviel/flux_redux_bfl", subfolder="image_encoder", torch_dtype=torch.float16).to("cuda").eval()

        # Load tokenizers
        self.tokenizer = LlamaTokenizerFast.from_pretrained(model_path, subfolder='tokenizer')
        self.tokenizer_2 = CLIPTokenizer.from_pretrained(model_path, subfolder='tokenizer_2')

        # Load text encoders
        self.text_encoder = LlamaModel.from_pretrained(model_path, subfolder='text_encoder', torch_dtype=torch.float16).to("cuda").eval()
        self.text_encoder_2 = CLIPTextModel.from_pretrained(model_path, subfolder='text_encoder_2', torch_dtype=torch.float16).to("cuda").eval()

        # Load transformer
        self.transformer = HunyuanVideoTransformer3DModelPacked.from_pretrained("lllyasviel/FramePackI2V_HY", torch_dtype=torch.bfloat16).to("cuda").eval()
        self.transformer.high_quality_fp32_output_for_inference = True

        # Load VAE
        vae_path = os.path.join(model_path, "vae")
        self.vae = AutoencoderKLHunyuanVideo.from_pretrained(vae_path, torch_dtype=torch.float16).to("cuda").eval()
        self.vae.enable_tiling()

        print("✅ Tous les composants sont prêts.")

    def predict(
        self,
        image: Path = Input(description="Image d'entrée (.jpg ou .png)"),
        prompt: str = Input(description="Prompt décrivant le mouvement"),
        seed: int = Input(default=42, description="Seed pour la reproductibilité"),
        steps: int = Input(default=25, description="Nombre d'étapes de sampling"),
        duration_seconds: float = Input(default=5.0, description="Durée de la vidéo en secondes"),
        fps: int = Input(default=30, description="Framerate de sortie"),
    ) -> Path:
        import numpy as np
        from diffusers_helper.hunyuan import encode_prompt_conds, vae_encode, vae_decode
        from diffusers_helper.utils import resize_and_center_crop, crop_or_pad_yield_mask
        from diffusers_helper.clip_vision import hf_clip_vision_encode
        from PIL import Image
        import tempfile
        import torch
        import einops
        import torchvision.io

        print(f"📸 Chargement de l’image depuis {image}")
        img = Image.open(image).convert("RGB")
        img_np = np.array(img)

        print("🧠 Encodage CLIP Vision...")
        image_features = hf_clip_vision_encode(img_np, self.image_processor, self.image_encoder)
        image_embeddings = image_features.last_hidden_state.to(torch.bfloat16)

        # Resize and center crop to 640 bucket
        height, width = 640, 640
        img_resized = resize_and_center_crop(img_np, target_width=width, target_height=height)

        # Torch image format: (B, C, T, H, W)
        img_tensor = torch.from_numpy(img_resized).float() / 127.5 - 1
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).unsqueeze(2).to(self.vae.device)

        print("🧠 Encodage VAE...")
        latents = vae_encode(img_tensor, self.vae)

        print("🧠 Encodage du prompt...")
        llama_vec, clip_pooler = encode_prompt_conds(prompt, self.text_encoder, self.text_encoder_2, self.tokenizer, self.tokenizer_2)
        llama_vec, llama_mask = crop_or_pad_yield_mask(llama_vec, length=512)

        # Fake negative prompt
        llama_vec_n = torch.zeros_like(llama_vec)
        clip_pooler_n = torch.zeros_like(clip_pooler)
        llama_mask_n = torch.ones_like(llama_mask)

        print("🎞️ Sampling...")
        generator = torch.Generator(device=self.transformer.device).manual_seed(seed)
        frames = int(duration_seconds * fps)
        num_latent_frames = frames + 3  # = latent_window_size * 4

        llama_vec = llama_vec.to(torch.bfloat16)
        clip_pooler = clip_pooler.to(torch.bfloat16)
        llama_vec_n = llama_vec_n.to(torch.bfloat16)
        clip_pooler_n = clip_pooler_n.to(torch.bfloat16)
        llama_mask = llama_mask.to(torch.bfloat16)
        llama_mask_n = llama_mask_n.to(torch.bfloat16)
        latents = latents.to(torch.bfloat16)

        output_latents = sample_hunyuan(
            transformer=self.transformer,
            sampler='unipc',
            width=width,
            height=height,
            frames=num_latent_frames,
            real_guidance_scale=1.0,
            distilled_guidance_scale=10.0,
            guidance_rescale=0.0,
            num_inference_steps=steps,
            generator=generator,
            prompt_embeds=llama_vec.to(self.transformer.device),
            prompt_embeds_mask=llama_mask.to(self.transformer.device),
            prompt_poolers=clip_pooler.to(self.transformer.device),
            negative_prompt_embeds=llama_vec_n.to(self.transformer.device),
            negative_prompt_embeds_mask=llama_mask_n.to(self.transformer.device),
            negative_prompt_poolers=clip_pooler_n.to(self.transformer.device),
            device=self.transformer.device,
            dtype=torch.bfloat16,
            image_embeddings=image_embeddings,
            latent_indices=None,
            clean_latents=latents.to(self.transformer.device),
            clean_latent_indices=None,
            clean_latents_2x=None,
            clean_latent_2x_indices=None,
            clean_latents_4x=None,
            clean_latent_4x_indices=None,
            callback=None,
        )

        print("🖼️ Décodage final...")
        decoded = vae_decode(output_latents, self.vae).cpu()  # [B, C, T, H, W]
        decoded = decoded[0]  # [C, T, H, W]
        video = einops.rearrange(decoded, 'c t h w -> t h w c')  # [T, H, W, C]
        video = (video * 255).clamp(0, 255).to(torch.uint8)

        output_path = tempfile.mktemp(suffix=".mp4")
        torchvision.io.write_video(output_path, video, fps=fps, video_codec='libx264', options={'crf': str(int(16))})
        print(f"🎬 Vidéo enregistrée à {output_path}")

        return Path(output_path)

