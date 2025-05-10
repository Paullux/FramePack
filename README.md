# 🎞️ FramePack Runner – Video Generation with Image + Prompt

[![Run on Replicate](https://replicate.com/badge.svg)](https://replicate.com/paullux/framepack-runner)
👉 [Try it on Replicate](https://replicate.com/paullux/framepack-runner)

**FramePack** is a next-frame video generation architecture developed by researchers at Stanford.
This repo provides a simple interface to use FramePack via [Replicate](https://replicate.com/paullux/framepack-runner), allowing you to generate short videos from a single image and a motion-focused prompt.

---

## 🚀 Try it now

Run the model on [Replicate](https://replicate.com/paullux/framepack-runner) directly, or from your terminal:

```bash
replicate run paullux/framepack-runner \
  -v image=@input.png \
  -v prompt="A cat jumps backward in surprise" \
  -v seed=123 \
  -v steps=30 \
  -v duration_seconds=5 \
  -v fps=30
```

---

## 🧠 About FramePack

> *Packing Input Frame Contexts in Next-Frame Prediction Models for Video Generation*
> *Lvmin Zhang, Maneesh Agrawala – Stanford University, 2025*

FramePack compresses temporal context into fixed-length representations, making it highly efficient for generating long video sequences.

- 📄 Project page

- 🧪 Uses the `FramePackPipeline` from 🤗 `diffusers`


---

## 📦 Inputs

| Name               | Type      | Description                                     |
|--------------------|-----------|-------------------------------------------------|
| `image`            | `file`    | The input image (`.png` or `.jpg`)              |
| `prompt`           | `string`  | A motion-focused description prompt             |
| `seed`             | `integer` | Random seed for reproducibility (default: `42`) |
| `steps`            | `integer` | Sampling steps (default: `25`)                  |
| `duration_seconds` | `number`  | Duration of the video in seconds (default: `5`) |
| `fps`              | `integer` | Output video frame rate (default: `30`)         |

---

## 📤 Output

Returns an `.mp4` video file composed of all generated frames.


---

## 🛠 How it works

Internally, FramePack uses:

- Text encoding (`LLaMA`, `CLIP`)
- Vision encoding (`SigLIP`)
- VAE for latent transformation
- Transformer for temporal prediction

The video is generated via next-frame sampling, decoded, and saved as an `.mp4`.

## 📸 Example Prompt Ideas

- The robot jumps forward and transforms mid-air.

- A woman spins slowly in a dark room, lit by candlelight.

- The camera zooms toward a cat staring at a moving shadow.


---

## 📃 License
This implementation uses FramePack under its original academic license.
Your usage of Replicate infrastructure is bound by their terms of service.


---

## 🙌 Credits

Based on [FramePack](https://github.com/lllyasviel/FramePack/) by Lvmin Zhang & Maneesh Agrawala (Stanford).


