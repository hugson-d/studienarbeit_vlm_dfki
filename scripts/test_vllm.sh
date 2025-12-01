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
# Vorher: sbatch scripts/setup_venv.sh
# ============================================

# Projekt-Root = SLURM_SUBMIT_DIR (wo sbatch aufgerufen wurde)
PROJECT_ROOT="${SLURM_SUBMIT_DIR}"

# Falls aus scripts/ gestartet, eine Ebene hoch
if [[ "$(basename $PROJECT_ROOT)" == "scripts" ]]; then
    PROJECT_ROOT="$(dirname $PROJECT_ROOT)"
fi

# WICHTIG: venv auf /netscratch (nicht $HOME oder Projekt-Root!)
VENV_PATH="/netscratch/$USER/vlm_venv"

echo "=========================================="
echo "PROJECT_ROOT: $PROJECT_ROOT"
echo "=========================================="

# Prüfen ob Projekt existiert
if [[ ! -f "$PROJECT_ROOT/dataset_final.json" ]]; then
    echo "❌ FEHLER: dataset_final.json nicht gefunden in $PROJECT_ROOT"
    exit 1
fi

# Prüfen ob venv existiert
if [[ ! -d "$VENV_PATH" ]]; then
    echo "❌ FEHLER: venv nicht gefunden in $VENV_PATH"
    echo "Bitte erst ausführen: sbatch scripts/setup_venv.sh"
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
    
    # Virtual Environment aktivieren
    echo '🐍 Aktiviere .venv...'
    source $VENV_PATH/bin/activate
    echo 'Python:' \$(which python)
    
    # vLLM nachinstallieren falls nicht vorhanden
    pip show vllm > /dev/null 2>&1 || pip install --quiet vllm
    
    echo ''
    echo '🏃 Starte Test...'
    python $PROJECT_ROOT/src/eval/test_vllm_qwen.py
    
    echo ''
    echo '=========================================='
    echo '✅ Fertig:' \$(date)
    echo '=========================================='
  "
