#!/bin/bash
#SBATCH --job-name=vlm_qwen_vllm
#SBATCH --partition=A100-40GB,RTXA6000,RTXA6000-SLT,L40S,H100-SLT-NP,H100,H200,A100-80GB,H100-SLT,A100-PCI,H200-AV,H200-DA,H200-PCI,H200-SDS
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

# ------------------------------
# 1. Pfade und Umgebung
# ------------------------------
PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
if [[ "$(basename "$PROJECT_ROOT")" == "scripts" ]]; then
    PROJECT_ROOT="$(dirname "$PROJECT_ROOT")"
fi

# Validierung
if [[ ! -f "$PROJECT_ROOT/dataset_final.json" ]]; then
    echo "❌ dataset_final.json nicht gefunden. Root: $PROJECT_ROOT"
    exit 1
fi

# Caches auf /netscratch
export PIP_CACHE_DIR="/netscratch/$USER/.cache/pip"
export HF_HOME="/netscratch/$USER/.cache/huggingface"
# vLLM spezifischer Cache (optional, nutzt meist HF_HOME)
export VLLM_CACHE_ROOT="/netscratch/$USER/.cache/vllm"
mkdir -p "$PIP_CACHE_DIR" "$HF_HOME" "$VLLM_CACHE_ROOT"

# HF Token laden
for SECRET_FILE in "$PROJECT_ROOT/.env" "$HOME/.hf_token"; do
    if [[ -f "$SECRET_FILE" ]]; then
        set -a
        source "$SECRET_FILE"
        set +a
        break
    fi
done

if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "⚠️ HF_TOKEN fehlt. Qwen2.5 erfordert Authentifizierung!"
    exit 1
fi

export VLM_PROJECT_ROOT="$PROJECT_ROOT"
export PYTHONUNBUFFERED=1

# ------------------------------
# 2. vLLM Spezifische Umgebung
# ------------------------------
# Verhindert Deadlocks in manchen SLURM/Container Umgebungen
export VLLM_WORKER_MULTIPROC_METHOD=spawn
# Deaktiviert Telemetrie
export VLLM_NO_USAGE_STATS=1
# FlashInfer Workspace (falls FlashInfer genutzt wird, für Performance)
export TORCH_EXTENSIONS_DIR="/netscratch/$USER/.cache/torch_extensions"
mkdir -p "$TORCH_EXTENSIONS_DIR"

echo "=========================================="
echo "🚀 VLM Benchmark: Qwen2.5-VL-3B (vLLM)"
echo "PROJECT_ROOT: $PROJECT_ROOT"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "=========================================="

# Annahme: Der Python-Code von vorhin liegt unter src/eval/models/benchmark_vllm.py
SCRIPT_PATH="$PROJECT_ROOT/src/eval/models/benchmark_vllm.py"

# ------------------------------
# 3. Ausführung
# ------------------------------
# WICHTIG: Deine install.sh muss 'pip install vllm qwen-vl-utils pandas' enthalten!

srun \
    --container-image=/enroot/nvcr.io_nvidia_pytorch_23.12-py3.sqsh \
    --container-mounts=/netscratch:/netscratch,/ds:/ds:ro,"$PROJECT_ROOT":"$PROJECT_ROOT" \
    --container-workdir="$PROJECT_ROOT" \
    --task-prolog="$PROJECT_ROOT/scripts/install_ide.sh" \
    python "$SCRIPT_PATH"

echo "✅ Job abgeschlossen"