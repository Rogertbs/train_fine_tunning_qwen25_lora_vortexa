#!/bin/bash
# Copie para env.sh e ajuste a GPU se necessario.
export LAB="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export UV_CACHE_DIR="$LAB/.uv-cache"
export HF_HOME="$LAB/models/hf-cache"

# Use 0 se a ordem CUDA da sua maquina for convencional.
# Para selecionar uma GPU especifica, prefira o UUID retornado por nvidia-smi.
# export CUDA_VISIBLE_DEVICES=0

export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
