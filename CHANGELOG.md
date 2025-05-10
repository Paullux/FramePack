## [2.0.0] - 2025-05-10

### 🚀 Nouveautés
- Support complet de FramePack avec image + prompt (text + vision embedding)
- Génération vidéo stable avec VAE, encodage CLIP Vision, et sample_hunyuan
- Export `.mp4` avec `torchvision.io.write_video`

### ✅ Corrigé
- Problème de dtype (Half vs BFloat16)
- Crash lors de l'absence de `image_embeddings`
- Rearrange avec `einops` corrigé

### 🔧 Dépendances clés
- `diffusers==0.33.1`
- `torch==2.1.2`
- `flash-attention==2.7.3`
- `accelerate==1.6.0`
