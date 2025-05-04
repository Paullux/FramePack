# predict.py
import cog
import torch
from PIL import Image
from diffusers import FramePackPipeline

class Predictor(cog.BasePredictor):
    def setup(self):
        print("📦 Loading model...")
        self.pipe = FramePackPipeline.from_pretrained(
            "githubcto/framepack",
            torch_dtype=torch.float16
        )
        self.pipe.to("cuda")

    @cog.input("image", type=Path, help="Input image (.jpg or .png)")
    @cog.input("prompt", type=str, default="The man dances powerfully, full of energy.")
    @cog.input("frames", type=int, default=60)
    @cog.input("fps", type=int, default=24)
    def predict(self, image, prompt, frames, fps) -> Path:
        print("🖼️ Loading image...")
        image = Image.open(image).convert("RGB")

        print("🎥 Generating frames...")
        video_frames = self.pipe(prompt=prompt, image=image, num_frames=frames).frames

        print("💾 Saving frames...")
        os.makedirs("frames", exist_ok=True)
        for i, frame in enumerate(video_frames):
            frame.save(f"frames/{i:04d}.png")

        print("🧬 Encoding video with ffmpeg...")
        output_path = "out.mp4"
        os.system(f"ffmpeg -y -framerate {fps} -i frames/%04d.png -c:v libx264 -pix_fmt yuv420p {output_path}")

        print("✅ Done.")
        return Path(output_path)
