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
  -v prompt="The man dances powerfully, full of energy." \
  -v frames=60 \
  -v fps=24
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

| Name   | Type   | Description                             |
|--------|--------|-----------------------------------------|
| `image`  | `file`   | The input image (.png or .jpg)          |
| `prompt` | `string` | A motion-focused description prompt     |
| `frames` | `integer` | Number of frames to generate (default: `60`) |
| `fps`    | `integer` | Output video frame rate (default: `24`)  |


---

## 📤 Output

Returns an `.mp4` video file composed of all generated frames.


---

## 🛠 How it works

```bash
python framepack_runner.py \
  --input input.png \
  --prompt "The man runs in slow motion through heavy rain" \
  --frames 60 \
  --fps 24
```
All frames are saved locally and then compiled into a video using `ffmpeg`.

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


