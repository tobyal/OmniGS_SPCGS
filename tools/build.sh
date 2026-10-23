#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
OD=/mnt/sda/ubuntu/lkx/od
export CUDA_HOME=/usr/local/cuda-11.8 CUDACXX=/usr/local/cuda-11.8/bin/nvcc
export PATH="/mnt/sda/ubuntu/anaconda3/envs/lkx_omnigs/bin:$CUDA_HOME/bin:$PATH"
cmake -S "$ROOT/code" -B "$ROOT/code/build" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86 -DTorch_DIR="$OD/deps/libtorch/share/cmake/Torch" -DOpenCV_DIR="$OD/deps/opencv-install/lib/cmake/opencv4" -DCUDA_TOOLKIT_ROOT_DIR="$CUDA_HOME" -Dglm_DIR="$OD/deps/glm-install/lib/cmake/glm"
cmake --build "$ROOT/code/build" -j 4 --target spcgs_native
