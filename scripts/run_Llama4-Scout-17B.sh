#!/bin/bash
#SBATCH --job-name=vlm_llama4_scout_17b
#SBATCH --partition=H100,H200,A100-80GB,H100-SLT,A100-PCI,H200-AV,H200-PCI
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

# ------------------------------
# Pfade und Umgebungsvariablen
# ------------------------------
PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
if [[ "$(basename "$PROJECT_ROOT")" == "scripts" ]]; then
    PROJECT_ROOT="$(dirname "$PROJECT_ROOT")"
fi

if [[ ! -f "$PROJECT_ROOT/dataset_final.json" ]]; then
    echo "dataset_final.json nicht gefunden. Bitte aus dem Repo-Root starten."
    exit 1
fi

# Caches auf /netscratch (schneller + mehr Platz)
export PIP_CACHE_DIR="/netscratch/$USER/.cache/pip"
export HF_HOME="/netscratch/$USER/.cache/huggingface"
mkdir -p "$PIP_CACHE_DIR" "$HF_HOME"

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
    echo "HF_TOKEN nicht gesetzt. Gated Modelle werden fehlschlagen."
else
    echo "HF_TOKEN geladen"
fi

export VLM_PROJECT_ROOT="$PROJECT_ROOT"
export PYTHONUNBUFFERED=1

echo "=========================================="
echo "🚀 VLM Benchmark: Llama4-Scout-17B"
echo "PROJECT_ROOT: $PROJECT_ROOT"
echo "=========================================="

# ------------------------------
# Container mit venv + Installation starten
# ------------------------------
srun \
    --container-image=/enroot/nvcr.io_nvidia_pytorch_23.12-py3.sqsh \
    --container-mounts=/netscratch:/netscratch,/ds:/ds:ro,"$PROJECT_ROOT":"$PROJECT_ROOT" \
    --container-workdir="$PROJECT_ROOT" \
    bash -c '
        echo "📦 Erstelle venv und installiere Dependencies..."
        # Venv erstellen (falls nicht vorhanden)
        VENV_PATH="/netscratch/$USER/.venv/llama4_scout"
        if [[ ! -d "$VENV_PATH" ]]; then
            python -m venv "$VENV_PATH"
            echo "✅ Venv erstellt: $VENV_PATH"
        fi
        # Venv aktivieren
        source "$VENV_PATH/bin/activate"
        # Dependencies installieren
        pip install --upgrade pip
        # WICHTIG: Alte torchvision aus Container isolieren (nms operator error)
        pip uninstall -y torchvision 2>/dev/null || true
        # WICHTIG: numpy<2.0, bitsandbytes für 4-bit Quantisierung
        # transformers>=4.51.0 für AutoModelForVision2Seq Support
        pip install "numpy<2.0" "transformers>=4.51.0" "accelerate>=0.33.0" "huggingface_hub>=0.24.0" "pydantic>=2.0" "python-dotenv>=1.0" "pandas" "openpyxl>=3.1" "tqdm" "pillow>=10.0" "safetensors>=0.4.0" "torch>=2.0" "torchvision>=0.15.0" "bitsandbytes>=0.41.0"
        echo "✅ Installation abgeschlossen"
        echo "DEBUG: Python: $(which python)"
        python -c "import transformers; print(f\"transformers: {transformers.__version__}\")"
        python -c "import bitsandbytes; print(f\"bitsandbytes: {bitsandbytes.__version__}\")"
        python -c "import torchvision; print(f\"torchvision: {torchvision.__version__}\")"
        # Python-Skript ausführen
        python '"$PROJECT_ROOT"'/src/eval/models/run_Llama4-Scout-17B.py
    '

echo "✅ Job abgeschlossen"
