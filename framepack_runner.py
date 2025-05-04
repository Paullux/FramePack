import torch
from diffusers import FramePackPipeline
from PIL import Image
import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to a .png or .jpg image")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt")
    parser.add_argument("--frames", type=int, default=60, help="NNumber of frames to generate")
    parser.add_argument("--fps", type=int, default=24, help="Frames per second for the final video")
    parser.add_argument("--output", type=str, default="out.mp4", help="Name of the output video")
    args = parser.parse_args()

    print("📦 Loading the FramePack template...")
    pipe = FramePackPipeline.from_pretrained("githubcto/framepack", torch_dtype=torch.float16)
    pipe.to("cuda")

    print("🖼️ Loading image...")
    image = Image.open(args.input).convert("RGB")

    print("🎥 Generation of frames...")
    video_frames = pipe(prompt=args.prompt, image=image, num_frames=args.frames).frames

    print("💾 Saving frames...")
    os.makedirs("frames", exist_ok=True)
    for i, frame in enumerate(video_frames):
        frame.save(f"frames/{i:04d}.png")

    print("🧬 Compilation in .mp4 with ffmpeg...")
    os.system(f"ffmpeg -y -framerate {args.fps} -i frames/%04d.png -c:v libx264 -pix_fmt yuv420p {args.output}")

    print(f"✅ Generated video : {args.output}")

if __name__ == "__main__":
    main()
