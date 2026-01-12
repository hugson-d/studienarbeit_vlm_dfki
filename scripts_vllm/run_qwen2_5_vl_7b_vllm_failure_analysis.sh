#!/bin/bash
#SBATCH --job-name=vlm_qwen2_5_vl_7b_failure_analysis
#SBATCH --partition=H100,H200,A100-80GB,H100-SLT,A100-PCI
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

# ------------------------------
# Pfade und Umgebungsvariablen
# ------------------------------
PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
if [[ "$(basename "$PROJECT_ROOT")" == "scripts_vllm" ]]; then
    PROJECT_ROOT="$(dirname "$PROJECT_ROOT")"
fi

export PIP_CACHE_DIR="/netscratch/$USER/.cache/pip"
export HF_HOME="/netscratch/$USER/.cache/huggingface"
mkdir -p "$PIP_CACHE_DIR" "$HF_HOME"

if [[ -f "$PROJECT_ROOT/.env" ]]; then
    set -a; source "$PROJECT_ROOT/.env"; set +a
fi

export VLM_PROJECT_ROOT="$PROJECT_ROOT"
export PROJECT_ROOT
export PYTHONUNBUFFERED=1

# DEAKTIVIERUNG VON FLASH ATTENTION
export VLLM_ATTENTION_BACKEND=XFORMERS

echo "=========================================="
echo "🔬 VLM Failure Analysis: Qwen2.5-VL-7B (NO FLASH ATTN)"
echo "Container: nvcr.io/nvidia/pytorch:23.12-py3"
echo "=========================================="

srun \
    --container-image=/enroot/nvcr.io_nvidia_pytorch_23.12-py3.sqsh \
    --container-mounts=/netscratch:/netscratch,/ds:/ds:ro,"$PROJECT_ROOT":"$PROJECT_ROOT" \
    --container-workdir="$PROJECT_ROOT" \
    bash -c '
        unset PYTHONPATH
        VENV_PATH="/netscratch/$USER/.venv/vllm_qwen_25_failure_analysis"
        
        if [ -d "$VENV_PATH" ]; then
            echo "🧹 Lösche alten venv..."
            rm -rf "$VENV_PATH"
        fi
        
        echo "📦 Erstelle neuen venv..."
        python -m venv "$VENV_PATH"
        source "$VENV_PATH/bin/activate"
        pip install --upgrade pip
        
        # Installation von xformers als Flash-Attention Ersatz
        pip install "vllm>=0.6.3" xformers xgrammar pydantic pandas tqdm qwen-vl-utils

        export LD_LIBRARY_PATH=$(python -c "import torch; print(torch._C.__file__)" | xargs dirname):$LD_LIBRARY_PATH
        export VLLM_ATTENTION_BACKEND=XFORMERS
        
        python "$PROJECT_ROOT/src/eval/vllm_models/run_qwen2_5_vl_7b_vllm_failure_analysis.py"
    '

echo "✅ Fertig!"