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

# Projektpfad ermitteln
PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
if [[ "$(basename "$PROJECT_ROOT")" == "scripts" ]]; then
    PROJECT_ROOT="$(dirname "$PROJECT_ROOT")"
fi

# WICHTIG: Cache auf Netscratch legen (Home ist zu klein/langsam)
export HF_HOME="/netscratch/$USER/.cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME"
mkdir -p "$HF_HOME"

# HF Token laden
if [[ -f "$PROJECT_ROOT/.env" ]]; then
    set -a; source "$PROJECT_ROOT/.env"; set +a
fi

export VLM_PROJECT_ROOT="$PROJECT_ROOT"
export PYTHONUNBUFFERED=1

echo "=== [JOB] Starte Ovis2.5-9B auf $(hostname) ==="
echo "Cache Dir: $HF_HOME"

srun \
    --container-image=/enroot/nvcr.io_nvidia_pytorch_23.12-py3.sqsh \
    --container-mounts=/netscratch:/netscratch,"$PROJECT_ROOT":"$PROJECT_ROOT" \
    --container-workdir="$PROJECT_ROOT" \
    --task-prolog="$PROJECT_ROOT/scripts/install_ovis_9b_test.sh" \
    python "$PROJECT_ROOT/src/eval/models/test.py"

echo "=== [JOB] Beendet ==="