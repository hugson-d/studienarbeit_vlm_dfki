#!/bin/bash
#SBATCH --job-name=vlm_internvl3_8b
#SBATCH --partition=H100
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# ============================================
# WICHTIG: Job aus Projekt-Root starten mit:
#   sbatch scripts/slurm_xxx.sh
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
    echo "Bitte starte den Job aus dem Projekt-Root: sbatch scripts/slurm_internvl3_8b.sh"
    exit 1
fi

# Logs-Verzeichnis erstellen
mkdir -p "$PROJECT_ROOT/evaluation_results/logs"

# Outputs verschieben nach Job-Ende
trap "mv ${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out $PROJECT_ROOT/evaluation_results/logs/ 2>/dev/null; mv ${SLURM_JOB_NAME}_${SLURM_JOB_ID}.err $PROJECT_ROOT/evaluation_results/logs/ 2>/dev/null" EXIT

# HF_TOKEN aus .env laden
if [ -f "$PROJECT_ROOT/.env" ]; then
    set -a
    source "$PROJECT_ROOT/.env"
    set +a
fi

# Debug: Token prüfen
if [ -z "$HF_TOKEN" ]; then
    echo "⚠️ WARNUNG: HF_TOKEN nicht gesetzt! Gated Models werden fehlschlagen."
    echo "   Erstelle .env Datei mit: HF_TOKEN=hf_xxx..."
else
    echo "✅ HF_TOKEN geladen"
fi

# Projekt-Root als Umgebungsvariable für Python-Skripte
export VLM_PROJECT_ROOT="$PROJECT_ROOT"
export HF_TOKEN

# Container starten
srun \
  --container-image=/enroot/nvcr.io_nvidia_pytorch_23.12-py3.sqsh \
  --container-mounts=/netscratch:/netscratch,/ds:/ds:ro,"$PROJECT_ROOT":"$PROJECT_ROOT" \
  --export=ALL,HF_TOKEN,VLM_PROJECT_ROOT \
  bash -c "
    echo '=========================================='
    echo '🚀 VLM Benchmark: InternVL3-8B'
    echo '=========================================='
    echo 'Start:' \$(date)
    echo 'HF_TOKEN: '\${HF_TOKEN:+gesetzt}
    echo 'GPU:'
    nvidia-smi --query-gpu=name,memory.total --format=csv
    echo ''
    
    # In Projektverzeichnis wechseln
    cd $PROJECT_ROOT
    echo 'Working Directory:' \$(pwd)
    
    # Python-Pakete installieren (kompatibel mit PyTorch 2.2.0 im Container)
    echo '📦 Installiere Python-Pakete...'
    pip install --quiet --no-warn-script-location \
      'transformers==4.44.2' \
      'accelerate==0.33.0' \
      'bitsandbytes==0.43.3' \
      'pillow>=10.0.0' \
      'pydantic>=2.0.0' \
      'pandas<1.6' \
      'openpyxl>=3.1.0' \
      'python-dotenv>=1.0.0' \
      'huggingface_hub==0.24.7' \
      'tqdm>=4.66.0' \
      'safetensors>=0.4.0' \
      'tokenizers>=0.19.0' \
      2>&1 | grep -v 'dependency resolver' | grep -v 'incompatible' || true
    
    # Flash Attention (optional)
    pip install --quiet flash-attn --no-build-isolation 2>/dev/null || echo '⚠️ Flash Attention nicht installiert (optional)'
    
    echo ''
    echo '🏃 Starte Benchmark...'
    python $PROJECT_ROOT/src/eval/models/run_internvl3_8b.py
    
    echo ''
    echo '=========================================='
    echo '✅ Fertig:' \$(date)
    echo '=========================================='
  "
