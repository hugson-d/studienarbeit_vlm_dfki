#!/bin/bash
#SBATCH --job-name=ovis_9b_final
#SBATCH --partition=H100,H200,A100-80GB,H100-SLT,A100-PCI,H200-AV,H200-PCI
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

# Projektpfad setup
PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
if [[ "$(basename "$PROJECT_ROOT")" == "scripts" ]]; then
    PROJECT_ROOT="$(dirname "$PROJECT_ROOT")"
fi

# Environment Variablen
export HF_HOME="/netscratch/$USER/.cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_HUB_ENABLE_HF_TRANSFER=1
mkdir -p "$HF_HOME"

# Token laden
if [[ -f "$PROJECT_ROOT/.env" ]]; then
    set -a; source "$PROJECT_ROOT/.env"; set +a
fi

export VLM_PROJECT_ROOT="$PROJECT_ROOT"
export PYTHONUNBUFFERED=1

echo "=========================================="
echo "🚀 Start Job: Ovis2.5-9B (Wrapper Mode)"
echo "PROJECT_ROOT: $PROJECT_ROOT"
echo "=========================================="

# WICHTIG: Kein --task-prolog mehr!
# Wir rufen direkt 'bash scripts/entrypoint_ovis.sh' im Container auf.

srun \
    --container-image=/enroot/nvcr.io_nvidia_pytorch_23.12-py3.sqsh \
    --container-mounts=/netscratch:/netscratch,"$PROJECT_ROOT":"$PROJECT_ROOT" \
    --container-workdir="$PROJECT_ROOT" \
    bash "$PROJECT_ROOT/scripts/entrypoint_ovis.sh"

echo "✅ srun beendet"