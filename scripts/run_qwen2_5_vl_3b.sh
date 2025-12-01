#!/bin/bash
#SBATCH --job-name=vlm_qwen2_5_vl_3b
#SBATCH --partition=H100
#SBATCH --gpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

# ------------------------------
# Pfade und Umgebungsvariablen
# ------------------------------
PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
if [[ "$(basename "$PROJECT_ROOT")" == "scripts" ]]; then
  PROJECT_ROOT="$(dirname "$PROJECT_ROOT")"
fi

if [[ ! -f "$PROJECT_ROOT/dataset_final.json" ]]; then
  echo "❌ dataset_final.json nicht im Projektverzeichnis gefunden. Bitte aus dem Repo-Root starten."
  exit 1
fi

# Caches und Venv liegen auf /netscratch
export PIP_CACHE_DIR="/netscratch/$USER/.cache/pip"
export HF_HOME="/netscratch/$USER/.cache/huggingface"
export TRANSFORMERS_CACHE="/netscratch/$USER/.cache/huggingface/transformers"
export HF_DATASETS_CACHE="/netscratch/$USER/.cache/huggingface/datasets"
VENV_PATH="/netscratch/$USER/vlm_qwen2_5_vl_3b"

mkdir -p "$PIP_CACHE_DIR" "$HF_HOME" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE"

# HF Token laden (.env im Projekt hat Priorität)
for SECRET_FILE in "$PROJECT_ROOT/.env" "$PROJECT_ROOT/secrets.sh" "$HOME/.hf_token"; do
  if [[ -f "$SECRET_FILE" ]]; then
    set -a
    source "$SECRET_FILE"
    set +a
    break
  fi
done

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "⚠️ HF_TOKEN nicht gesetzt. Gated Modelle schlagen fehl."
else
  echo "✅ HF_TOKEN gefunden."
fi

export VLM_PROJECT_ROOT="$PROJECT_ROOT"
export HF_TOKEN
export PROJECT_ROOT
export VENV_PATH
export PYTHONUNBUFFERED=1

# ------------------------------
# Container starten und Benchmark ausführen
# ------------------------------
srun \
  --container-image=/enroot/nvcr.io_nvidia_pytorch_23.12-py3.sqsh \
  --container-mounts=/netscratch:/netscratch,/ds:/ds:ro,"$PROJECT_ROOT":"$PROJECT_ROOT" \
  --export=ALL,HF_TOKEN,VLM_PROJECT_ROOT,PROJECT_ROOT,VENV_PATH,PIP_CACHE_DIR,HF_HOME,TRANSFORMERS_CACHE,HF_DATASETS_CACHE \
  bash -c '
    set -euo pipefail
    echo "=========================================="
    echo "🚀 Starte Qwen2.5-VL-3B Benchmark"
    echo "PROJECT_ROOT: $PROJECT_ROOT"
    echo "VENV_PATH: $VENV_PATH"
    echo "=========================================="

    cd "$PROJECT_ROOT"

    # Venv anlegen, falls nicht vorhanden (Python 3.10 im Container)
    if [[ ! -d "$VENV_PATH" ]]; then
      echo "🐍 Erstelle venv unter $VENV_PATH"
      python3 -m venv "$VENV_PATH"
      source "$VENV_PATH/bin/activate"
      python -m pip install --upgrade pip
      # Torch Build passend zur H100/H200 (CUDA 12.1)
      python -m pip install --extra-index-url https://download.pytorch.org/whl/cu121 \
        "torch==2.3.1" "torchvision==0.18.1" "torchaudio==2.3.1"
      python -m pip install \
        "transformers>=4.44.0" \
        "accelerate>=0.33.0" \
        "huggingface_hub>=0.24.0" \
        "qwen-vl-utils>=0.0.9" \
        "pydantic>=2.6" \
        "python-dotenv>=1.0" \
        "pandas>=2.2" \
        "openpyxl>=3.1" \
        "tqdm" \
        "pillow>=10.0"
    else
      echo "🐍 Nutze existierende venv"
      source "$VENV_PATH/bin/activate"
    fi

    echo "Python: $(which python)"
    echo "Torch: $(python - <<"PY"
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
PY
)"

    echo "🏃 Laufen lassen..."
    python "$PROJECT_ROOT/src/eval/models/run_qwen2_5_vl_3b.py"

    echo "✅ Fertig"
  '
