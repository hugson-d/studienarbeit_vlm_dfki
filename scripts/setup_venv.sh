#!/bin/bash
#SBATCH --job-name=vlm_setup_venv
#SBATCH --partition=H100
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
if [[ "$(basename "$PROJECT_ROOT")" == "scripts" ]]; then
  PROJECT_ROOT="$(dirname "$PROJECT_ROOT")"
fi

VENV_PATH="/netscratch/$USER/vlm_venv"
PIP_CACHE_DIR="/netscratch/$USER/.cache/pip"
HF_CACHE_DIR="/netscratch/$USER/.cache/huggingface"

mkdir -p "$PIP_CACHE_DIR" "$HF_CACHE_DIR"

export PIP_CACHE_DIR
export HF_HOME="$HF_CACHE_DIR"
export TRANSFORMERS_CACHE="$HF_CACHE_DIR/transformers"
export HF_DATASETS_CACHE="$HF_CACHE_DIR/datasets"

# Logs nach evaluation_results/logs verschieben
mkdir -p "$PROJECT_ROOT/evaluation_results/logs"
trap "mv ${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out $PROJECT_ROOT/evaluation_results/logs/ 2>/dev/null; mv ${SLURM_JOB_NAME}_${SLURM_JOB_ID}.err $PROJECT_ROOT/evaluation_results/logs/ 2>/dev/null" EXIT

srun \
  --container-image=/enroot/nvcr.io_nvidia_pytorch_23.12-py3.sqsh \
  --container-mounts=/netscratch:/netscratch,"$PROJECT_ROOT":"$PROJECT_ROOT" \
  --export=ALL,PIP_CACHE_DIR,HF_HOME,TRANSFORMERS_CACHE,HF_DATASETS_CACHE,VENV_PATH,PROJECT_ROOT \
  bash -c '
    set -euo pipefail

    echo "=========================================="
    echo "🚀 Einmaliges Setup: virtuelles Environment"
    echo "PROJECT_ROOT: $PROJECT_ROOT"
    echo "VENV_PATH: $VENV_PATH"
    echo "=========================================="

    if [[ -d "$VENV_PATH" ]]; then
      echo "ℹ️ Venv existiert bereits unter $VENV_PATH"
    else
      echo "🐍 Erstelle venv..."
      python3 -m venv "$VENV_PATH"
    fi

    source "$VENV_PATH/bin/activate"
    python -m pip install --upgrade pip

    echo "📦 Installiere Kern-Abhängigkeiten (CUDA 12.1 Build)"
    python -m pip install --extra-index-url https://download.pytorch.org/whl/cu121 \
      "torch==2.3.1" "torchvision==0.18.1" "torchaudio==2.3.1"

    echo "📦 Installiere VLM-Dependencies"
    python -m pip install \
      "transformers>=4.44.0" \
      "accelerate>=0.33.0" \
      "huggingface_hub>=0.24.0" \
      "bitsandbytes>=0.43.0" \
      "qwen-vl-utils>=0.0.9" \
      "pydantic>=2.6" \
      "pillow>=10.0" \
      "pandas>=2.2" \
      "openpyxl>=3.1" \
      "tqdm" \
      "sentencepiece"

    echo "📦 Installiere lokale Projekt-Abhängigkeiten"
    python -m pip install -e "$PROJECT_ROOT"

    echo "✅ Setup abgeschlossen"
  '
