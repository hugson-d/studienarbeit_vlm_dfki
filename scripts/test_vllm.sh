#!/bin/bash
#SBATCH --job-name=test_vllm_qwen
#SBATCH --partition=H100
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=1:00:00
#SBATCH --output=test_vllm_%j.out
#SBATCH --error=test_vllm_%j.err

# ============================================
# vLLM Test für Qwen2.5-VL-7B
# ============================================

# Projekt-Root = SLURM_SUBMIT_DIR (wo sbatch aufgerufen wurde)
PROJECT_ROOT="${SLURM_SUBMIT_DIR}"

# Falls aus scripts/ gestartet, eine Ebene hoch
if [[ "$(basename $PROJECT_ROOT)" == "scripts" ]]; then
    PROJECT_ROOT="$(dirname $PROJECT_ROOT)"
fi

echo "=========================================="
echo "PROJECT_ROOT: $PROJECT_ROOT"
echo "=========================================="

# Prüfen ob Projekt existiert
if [[ ! -f "$PROJECT_ROOT/dataset_final.json" ]]; then
    echo "❌ FEHLER: dataset_final.json nicht gefunden in $PROJECT_ROOT"
    exit 1
fi

# HF_TOKEN aus .env oder secrets.sh laden
for SECRET_FILE in "$PROJECT_ROOT/.env" "$PROJECT_ROOT/secrets.sh" "$HOME/.hf_token"; do
    if [ -f "$SECRET_FILE" ]; then
        set -a
        source "$SECRET_FILE"
        set +a
        break
    fi
done

if [ -z "$HF_TOKEN" ]; then
    echo "⚠️ WARNUNG: HF_TOKEN nicht gesetzt!"
else
    echo "✅ HF_TOKEN geladen"
fi

export VLM_PROJECT_ROOT="$PROJECT_ROOT"
export HF_TOKEN

# Container starten
srun \
  --container-image=/enroot/nvcr.io_nvidia_pytorch_23.12-py3.sqsh \
  --container-mounts=/netscratch:/netscratch,/ds:/ds:ro,"$PROJECT_ROOT":"$PROJECT_ROOT" \
  --export=ALL,HF_TOKEN,VLM_PROJECT_ROOT \
  bash -c "
    echo '=========================================='
    echo '🚀 vLLM Test: Qwen2.5-VL-7B'
    echo '=========================================='
    echo 'Start:' \$(date)
    echo 'GPU:'
    nvidia-smi --query-gpu=name,memory.total --format=csv
    echo ''
    
    cd $PROJECT_ROOT
    
    # vLLM und Dependencies installieren
    echo '📦 Installiere vLLM und Dependencies...'
    pip install --quiet --no-warn-script-location \
      vllm \
      pillow \
      python-dotenv \
      huggingface_hub \
      2>&1 | grep -v 'dependency resolver' | grep -v 'incompatible' || true
    
    echo ''
    echo '🏃 Starte Test...'
    python $PROJECT_ROOT/src/eval/test_vllm_qwen.py
    
    echo ''
    echo '=========================================='
    echo '✅ Fertig:' \$(date)
    echo '=========================================='
  "
