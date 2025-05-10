#!/bin/bash

set -e

BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
HF_DIR="$BASE_DIR/hf_download"
WWW_SYMLINK="/var/www/html/hf_download"

MODEL_ID="hunyuanvideo-community/HunyuanVideo"
FLUX_ID="lllyasviel/flux_redux_bfl"

FORCE_DL=true
[[ "$1" == "--check" ]] && FORCE_DL=false && echo "🔍 Mode --check : vérification uniquement, pas de téléchargement forcé."

export HF_HOME="$HF_DIR"
mkdir -p "$HF_HOME"

python3 - <<EOF
import os
import sys
import torch
import contextlib

# 🔕 Patchs pour désactiver les barres de progression de tqdm
import transformers.utils.logging
transformers.utils.logging.disable_progress_bar()

import builtins
builtins.tqdm = lambda *a, **kw: iter(a[0] if a else [])

from transformers import (
    CLIPTextModel, AutoTokenizer,
    LlamaModel, LlamaTokenizerFast,
    SiglipVisionModel, SiglipImageProcessor,
)
from diffusers_helper.models.hunyuan_video_packed import AutoencoderKLHunyuanVideo

MODEL_LOCAL_PATH = os.path.join(os.environ['HF_HOME'], 'hub', 'models--hunyuanvideo-community--HunyuanVideo', 'snapshots', 'e8c2aaa66fe3742a32c11a6766aecbf07c56e773')

print('📥 AutoencoderKLHunyuanVideo → vae')
try:
    AutoencoderKLHunyuanVideo.from_pretrained(
        MODEL_LOCAL_PATH,
        subfolder='vae',
        local_files_only=True,
        device_map="cpu"
    )
    print('✅ AutoencoderKLHunyuanVideo : OK')
except Exception as e:
    print('❌ VAE :', e)

def try_load(desc, cls, path, subfolder=None):
    print(f'📥 {desc} → {subfolder or ""}')
    try:
        cls.from_pretrained(path, subfolder=subfolder, local_files_only=True, device_map="cpu")
        print(f'✅ {desc} : OK')
    except Exception as e:
        print(f'❌ {desc} :', e)

try_load('LlamaModel', LlamaModel, MODEL_LOCAL_PATH, 'text_encoder')
try_load('LlamaTokenizerFast', LlamaTokenizerFast, MODEL_LOCAL_PATH, 'tokenizer')
try_load('CLIPTextModel', CLIPTextModel, MODEL_LOCAL_PATH, 'text_encoder_2')
try_load('AutoTokenizer (CLIP)', AutoTokenizer, MODEL_LOCAL_PATH, 'tokenizer_2')
try_load('SiglipVisionModel', SiglipVisionModel, '$FLUX_ID', 'image_encoder')
try_load('SiglipImageProcessor', SiglipImageProcessor, '$FLUX_ID', 'feature_extractor')
EOF

if [[ ! -L "$WWW_SYMLINK" ]]; then
    echo "🔗 Création du lien symbolique pour Apache..."
    sudo ln -sfn "$HF_DIR" "$WWW_SYMLINK"
    echo "✅ Lien symbolique vers : $WWW_SYMLINK"
else
    echo "✅ Lien symbolique déjà présent."
fi

IP_LOCAL=$(hostname -I | awk '{print $1}')
echo "🌐 Accès local Apache : http://$IP_LOCAL/hf_download/"
echo "✅ Script terminé."
