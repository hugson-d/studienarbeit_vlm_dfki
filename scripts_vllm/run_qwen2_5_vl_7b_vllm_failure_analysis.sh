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

# Caches auf /netscratch
export PIP_CACHE_DIR="/netscratch/$USER/.cache/pip"
export HF_HOME="/netscratch/$USER/.cache/huggingface"
mkdir -p "$PIP_CACHE_DIR" "$HF_HOME"

# HF Token laden
if [[ -f "$PROJECT_ROOT/.env" ]]; then
    set -a; source "$PROJECT_ROOT/.env"; set +a
fi

export VLM_PROJECT_ROOT="$PROJECT_ROOT"
export PROJECT_ROOT
export PYTHONUNBUFFERED=1

echo "=========================================="
echo "🔬 VLM Failure Analysis: Qwen2.5-VL-7B"
echo "Container: nvcr.io/nvidia/pytorch:24.10-py3"
echo "=========================================="

# ------------------------------
# Ausführung im Container
# ------------------------------
srun \
    --container-image=/enroot/nvcr.io_nvidia_pytorch_24.10-py3.sqsh \
    --container-mounts=/netscratch:/netscratch,/ds:/ds:ro,"$PROJECT_ROOT":"$PROJECT_ROOT" \
    --container-workdir="$PROJECT_ROOT" \
    bash -c '
        unset PYTHONPATH
        
        # Immer frischen venv nutzen um Konflikte auszuschließen
        VENV_PATH="/netscratch/$USER/.venv/vllm_qwen_25_failure_analysis"
        if [ -d "$VENV_PATH" ]; then
            echo "🧹 Lösche alten venv..."
            rm -rf "$VENV_PATH"
        fi
        
        echo "📦 Erstelle neuen venv..."
        python -m venv "$VENV_PATH"
        
        source "$VENV_PATH/bin/activate"
        pip install --upgrade pip
        
        # vLLM installiert kompatibles flash-attn/xformers automatisch
        # Wir zwingen keine spezifische Version um ABI Konflikte zu meiden
        pip install "vllm>=0.6.3" xgrammar pydantic pandas tqdm qwen-vl-utils xformers

        # FORCE XFORMERS BACKEND: Verhindert Nutzung von flash_attn
        export VLLM_ATTENTION_BACKEND=XFORMERS

        # Fix für LD_LIBRARY_PATH (stellt sicher, dass PyTorch-C++ Libs gefunden werden)
        export LD_LIBRARY_PATH=$(python -c "import torch; print(torch._C.__file__)" | xargs dirname):$LD_LIBRARY_PATH
        
        echo "✅ Setup bereit. Starte Analyse..."
        echo "📂 Working Directory: $(pwd)"
        echo "📂 Project Root: $PROJECT_ROOT"
        
        # Sicherstellen dass wir das Script finden
        if [ ! -f "$PROJECT_ROOT/src/eval/vllm_models/run_qwen2_5_vl_7b_vllm_failure_analysis.py" ]; then
            echo "❌ Script nicht gefunden unter: $PROJECT_ROOT/src/eval/vllm_models/run_qwen2_5_vl_7b_vllm_failure_analysis.py"
            exit 1
        fi
        
        python "$PROJECT_ROOT/src/eval/vllm_models/run_qwen2_5_vl_7b_vllm_failure_analysis.py"
    '

echo "🎉 Analyse abgeschlossen."