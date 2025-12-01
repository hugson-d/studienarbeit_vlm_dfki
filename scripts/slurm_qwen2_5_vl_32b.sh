#!/bin/bash
#SBATCH --job-name=vlm_qwen2_5_vl_32b
#SBATCH --partition=H100
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# ============================================
# WICHTIG: 
# 1. Erst einmalig: sbatch scripts/setup_venv.sh
# 2. Dann Jobs starten: sbatch scripts/slurm_xxx.sh
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
    echo "Bitte starte den Job aus dem Projekt-Root: sbatch scripts/slurm_qwen2_5_vl_32b.sh"
    exit 1
fi

# Prüfen ob .venv existiert
if [[ ! -d "$VENV_PATH" ]]; then
    echo "❌ FEHLER: venv nicht gefunden in $VENV_PATH"
    echo "Bitte erst ausführen: sbatch scripts/setup_venv.sh"
    exit 1
fi

# Logs-Verzeichnis erstellen
mkdir -p "$PROJECT_ROOT/evaluation_results/logs"

# Outputs verschieben nach Job-Ende
trap "mv ${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out $PROJECT_ROOT/evaluation_results/logs/ 2>/dev/null; mv ${SLURM_JOB_NAME}_${SLURM_JOB_ID}.err $PROJECT_ROOT/evaluation_results/logs/ 2>/dev/null" EXIT

# HF_TOKEN aus .env oder secrets.sh laden
for SECRET_FILE in "$PROJECT_ROOT/.env" "$PROJECT_ROOT/secrets.sh" "$HOME/.hf_token"; do
    if [ -f "$SECRET_FILE" ]; then
        set -a
        source "$SECRET_FILE"
        set +a
        break
    fi
done

# Debug: Token prüfen
if [ -z "$HF_TOKEN" ]; then
    echo "⚠️ WARNUNG: HF_TOKEN nicht gesetzt! Gated Models werden fehlschlagen."
else
    echo "✅ HF_TOKEN geladen"
fi

# Projekt-Root als Umgebungsvariable für Python-Skripte
export VLM_PROJECT_ROOT="$PROJECT_ROOT"
export HF_TOKEN
export PROJECT_ROOT
export VENV_PATH

# Container starten
srun \
  --container-image=/enroot/nvcr.io_nvidia_pytorch_23.12-py3.sqsh \
  --container-mounts=/netscratch:/netscratch,/ds:/ds:ro,"$PROJECT_ROOT":"$PROJECT_ROOT" \
  --export=ALL,HF_TOKEN,VLM_PROJECT_ROOT,PROJECT_ROOT,VENV_PATH \
  bash -c '
    echo "=========================================="
    echo "🚀 VLM Benchmark: Qwen2.5-VL-32B"
    echo "=========================================="
    echo "Start: $(date)"
    echo "HF_TOKEN: ${HF_TOKEN:+gesetzt}"
    echo "GPU:"
    nvidia-smi --query-gpu=name,memory.total --format=csv
    echo ""
    
    # In Projektverzeichnis wechseln
    cd "$PROJECT_ROOT"
    
    # Virtual Environment aktivieren
    echo "🐍 Aktiviere .venv..."
    if [ -f "$VENV_PATH/bin/activate" ]; then
        source "$VENV_PATH/bin/activate" || exit 1
    else
        echo "❌ FEHLER: venv nicht gefunden: $VENV_PATH"
        echo "Bitte sbatch scripts/setup_venv.sh ausführen und warten!"
        exit 1
    fi
    
    echo "Python: $(which python)"
    echo ""
    
    echo "🏃 Starte Benchmark..."
    python "$PROJECT_ROOT/src/eval/models/run_qwen2_5_vl_32b.py"
    
    echo ""
    echo "=========================================="
    echo "✅ Fertig: $(date)"
    echo "=========================================="
  '
