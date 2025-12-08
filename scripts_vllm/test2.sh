#!/bin/bash
#SBATCH --job-name=qwen3_vl_bench
#SBATCH --partition=H100,A100-80GB
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=04:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

# ----------------------------------------------------------------------------
# UMGEBUNG
# ----------------------------------------------------------------------------
export PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
export VLM_PROJECT_ROOT="$PROJECT_ROOT"
export PIP_CACHE_DIR="/netscratch/$USER/.cache/pip"
export HF_HOME="/netscratch/$USER/.cache/huggingface"
mkdir -p "$PIP_CACHE_DIR" "$HF_HOME"

if [[ -f "$PROJECT_ROOT/.env" ]]; then export $(grep -v '^#' "$PROJECT_ROOT/.env" | xargs); fi

echo "🚀 Starte Qwen3-VL-8B Benchmark"

# ----------------------------------------------------------------------------
# CONTAINER
# ----------------------------------------------------------------------------
srun \
    --container-image=/enroot/nvcr.io_nvidia_pytorch_23.12-py3.sqsh \
    --container-mounts=/netscratch:/netscratch,"$PROJECT_ROOT":"$PROJECT_ROOT" \
    --container-workdir="$PROJECT_ROOT" \
    bash -c '
    
    unset PYTHONPATH
    VENV_DIR="/netscratch/$USER/.venv/qwen3_vl_env"

    if [[ ! -d "$VENV_DIR" ]]; then
        echo "🧹 Erstelle Venv..."
        python -m venv "$VENV_DIR"
    fi
    source "$VENV_DIR/bin/activate"

    echo "⬇️ Installiere Pakete..."
    pip install --upgrade pip
    
    # 1. OpenCV Fix (Immer zuerst)
    pip install --force-reinstall --no-cache-dir "opencv-python-headless>=4.10.0"

    # 2. Qwen3 Requirements
    # qwen-vl-utils ist zwingend.
    # accelerate für schnelles Laden.
    pip install "qwen-vl-utils" "accelerate" "transformers>=4.48.0"

    # 3. vLLM (Muss neu sein für Qwen3 Support)
    # vLLM >= 0.11.0 wird für volle Qwen3 Unterstützung empfohlen
    pip install "vllm>=0.6.4" "numpy<2.0" pandas tqdm pydantic python-dotenv

    echo "🔍 Prüfe Versionen..."
    python -c "import vllm; print(f\"vLLM: {vllm.__version__}\")"

    SCRIPT_PATH="qwen3_eval.py"
    if [[ ! -f "$SCRIPT_PATH" ]]; then SCRIPT_PATH="src/eval/vllm_models/qwen3_eval.py"; fi
    
    echo "▶️ Starte $SCRIPT_PATH"
    "$VENV_DIR/bin/python3" "$SCRIPT_PATH"
    '

echo "✅ Job beendet."