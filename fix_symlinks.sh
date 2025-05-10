#!/bin/bash
cd hf_download/hub/models--hunyuanvideo-community--HunyuanVideo/snapshots/*/vae || exit 1

for f in config.json diffusion_pytorch_model.safetensors; do
    real_target=$(realpath "$f")
    rm "$f"
    cp "$real_target" .
done

echo "✅ Symlinks remplacés par de vrais fichiers."
