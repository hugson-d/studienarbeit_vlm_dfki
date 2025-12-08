#!/bin/bash
#SBATCH --job-name=vlm_pixtral
#SBATCH --partition=H100,H200,A100-80GB,H100-SLT,A100-PCI
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=04:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

# ----------------------------------------------------------------------------
# UMGEBUNG & PFADE
# ----------------------------------------------------------------------------
export PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
export VLM_PROJECT_ROOT="$PROJECT_ROOT"

# Cache Verzeichnisse auf schnellen Storage legen
export PIP_CACHE_DIR="/netscratch/$USER/.cache/pip"
export HF_HOME="/netscratch/$USER/.cache/huggingface"
mkdir -p "$PIP_CACHE_DIR" "$HF_HOME"

# HF Token laden
if [[ -f "$PROJECT_ROOT/.env" ]]; then
    export $(grep -v '^#' "$PROJECT_ROOT/.env" | xargs)
fi

echo "🚀 Starte Pixtral Benchmark"
echo "📂 Root: $PROJECT_ROOT"

# ----------------------------------------------------------------------------
# CONTAINER & AUSFÜHRUNG
# ----------------------------------------------------------------------------
srun \
    --container-image=/enroot/nvcr.io_nvidia_pytorch_23.12-py3.sqsh \
    --container-mounts=/netscratch:/netscratch,"$PROJECT_ROOT":"$PROJECT_ROOT" \
    --container-workdir="$PROJECT_ROOT" \
    bash -c '
    
    # 1. Virtual Environment erstellen (Isolierung)
    VENV_DIR="/netscratch/$USER/.venv/pixtral_env"
    if [[ ! -d "$VENV_DIR" ]]; then
        echo "📦 Erstelle venv..."
        python -m venv "$VENV_DIR"
    fi
    source "$VENV_DIR/bin/activate"

    # 2. Installation der Dependencies
    # WICHTIG: numpy<2.0 verhindert Konflikte zwischen Pandas/Torch/vLLM
    echo "⬇️ Installiere Pakete..."
    pip install --upgrade pip
    pip install "vllm>=0.6.3" "numpy<2.0" pandas tqdm pydantic python-dotenv

    # 3. Python Skript starten
    SCRIPT_PATH="'"$PROJECT_ROOT"'/src/eval/vllm_models/run_ovis2_5_9b_vllm.py"
    
    if [[ ! -f "$SCRIPT_PATH" ]]; then
        # Fallback falls Skript im Root liegt
        SCRIPT_PATH="run_pixtral_12b_vllm.py"
    fi

    echo "▶️ Führe Skript aus: $SCRIPT_PATH"
    python3 "$SCRIPT_PATH"
    '

echo "✅ Job beendet."