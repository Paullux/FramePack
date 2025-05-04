from cog import BasePredictor, Input, Path
from diffusers import FramePackPipeline
from PIL import Image
import torch
import os
import shutil

print("✅ FramePackPipeline importé avec succès")

class Predictor(BasePredictor):
    def setup(self):
        self.pipe = FramePackPipeline.from_pretrained(
            "githubcto/framepack",
            torch_dtype=torch.float16
        )
        self.pipe.to("cuda")

    def predict(
        self,
        image: Path = Input(description="Image d'entrée (.png ou .jpg)"),
        prompt: str = Input(description="Texte décrivant le mouvement", default="The man dances powerfully, full of energy."),
        frames: int = Input(description="Nombre de frames à générer", default=60),
        fps: int = Input(description="Images par seconde", default=24),
    ) -> Path:
        img = Image.open(image).convert("RGB")
        video_frames = self.pipe(prompt=prompt, image=img, num_frames=frames).frames

        # Crée un dossier temporaire pour les frames
        frame_dir = "frames"
        os.makedirs(frame_dir, exist_ok=True)

        for i, frame in enumerate(video_frames):
            frame.save(f"{frame_dir}/{i:04d}.png")

        out_path = "out.mp4"
        os.system(f"ffmpeg -y -framerate {fps} -i {frame_dir}/%04d.png -c:v libx264 -pix_fmt yuv420p {out_path}")

        # Nettoyage optionnel du dossier frames (facultatif si tu veux le garder)
        shutil.rmtree(frame_dir)

        return Path(out_path)
