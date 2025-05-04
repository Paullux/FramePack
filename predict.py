from cog import BasePredictor, Input, Path
from diffusers import FramePackPipeline
from PIL import Image
import torch

class Predictor(BasePredictor):
    def setup(self):
        self.pipe = FramePackPipeline.from_pretrained("githubcto/framepack", torch_dtype=torch.float16)
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
        out_path = "out.mp4"

        import os
        os.makedirs("frames", exist_ok=True)
        for i, frame in enumerate(video_frames):
            frame.save(f"frames/{i:04d}.png")

        os.system(f"ffmpeg -y -framerate {fps} -i frames/%04d.png -c:v libx264 -pix_fmt yuv420p {out_path}")
        return Path(out_path)
