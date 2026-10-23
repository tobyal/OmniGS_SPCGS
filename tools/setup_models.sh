#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$ROOT/weights/clip-vit-base-patch16" "$ROOT/vendor"
cd "$ROOT"
# Isolated Python overlay: never change another experiment's environment.
if [ ! -f .venv/bin/python ]; then /mnt/sda/ubuntu/anaconda3/envs/jxy_sam2/bin/python -m venv --system-site-packages .venv; fi
if [ ! -d vendor/sam2 ]; then
    git clone --depth 1 https://github.com/facebookresearch/sam2.git vendor/sam2
    git -C vendor/sam2 fetch --depth 1 origin 2b90b9f5ceec907a1c18123530e92e794ad901a4
    git -C vendor/sam2 checkout --detach 2b90b9f5ceec907a1c18123530e92e794ad901a4
fi
SAM2_BUILD_CUDA=0 .venv/bin/pip install --no-build-isolation --no-deps -e vendor/sam2
.venv/bin/pip install 'numpy==1.26.4' 'hydra-core>=1.3.2' 'iopath>=0.1.10' 'omegaconf>=2.3' 'transformers==4.46.3' scipy pillow plyfile scikit-image matplotlib lpips
for name in config.json preprocessor_config.json tokenizer_config.json special_tokens_map.json vocab.json merges.txt pytorch_model.bin; do
    test -s "weights/clip-vit-base-patch16/$name" || curl -fL --retry 3 --connect-timeout 20 "https://huggingface.co/openai/clip-vit-base-patch16/resolve/57c216476eefef5ab752ec549e440a49ae4ae5f3/$name" -o "weights/clip-vit-base-patch16/$name.part"
    if test -f "weights/clip-vit-base-patch16/$name.part"; then mv "weights/clip-vit-base-patch16/$name.part" "weights/clip-vit-base-patch16/$name"; fi
done
test -s weights/sam2.1_hiera_tiny.pt || curl -fL --retry 3 --connect-timeout 20 https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt -o weights/sam2.1_hiera_tiny.pt
git -C vendor/sam2 rev-parse HEAD > weights/sam2_commit.txt
.venv/bin/pip freeze > weights/python_packages.txt
