#!/bin/bash
#SBATCH --job-name=vlm_internvl3_38b
#SBATCH --partition=H100,H200,A100-80GB,H100-SLT,A100-PCI,H200-AV,H200-PCI
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=2
#SBATCH --time=24:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# InternVL3-38B - FP16 (knapp unter 40B, kein 4-Bit nötig)

set -euo pipefail

# ------------------------------
# Pfade und Umgebungsvariablen
# ------------------------------
PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
if [[ "$(basename "$PROJECT_ROOT")" == "scripts" ]]; then
    PROJECT_ROOT="$(dirname "$PROJECT_ROOT")"
fi

if [[ ! -f "$PROJECT_ROOT/dataset_final.json" ]]; then
    echo "❌ dataset_final.json nicht gefunden. Bitte aus dem Repo-Root starten."
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
    echo "⚠️ HF_TOKEN nicht gesetzt. Gated Modelle werden fehlschlagen."
else
    echo "✅ HF_TOKEN geladen"
fi

export VLM_PROJECT_ROOT="$PROJECT_ROOT"
export PYTHONUNBUFFERED=1

echo "=========================================="
echo "🚀 VLM Benchmark: InternVL3-38B"
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
        VENV_PATH="/netscratch/$USER/.venv/internvl3"
        if [[ ! -d "$VENV_PATH" ]]; then
            python -m venv "$VENV_PATH"
            echo "✅ Venv erstellt: $VENV_PATH"
        fi
        # Venv aktivieren
        source "$VENV_PATH/bin/activate"
        # Dependencies installieren
        pip install --upgrade pip
        pip install "numpy<2.0" "transformers>=4.37.2" "accelerate>=0.33.0" "huggingface_hub>=0.24.0" "timm>=0.9.16" "pydantic>=2.0" "python-dotenv>=1.0" "pandas" "openpyxl>=3.1" "tqdm" "pillow>=10.0" "safetensors>=0.4.0" "torch>=2.0" "torchvision>=0.15.0"
        # Flash Attention 2 (optional, aber empfohlen)
        pip install "flash-attn>=2.3.0" --no-build-isolation || echo "⚠️ Flash Attention installation failed, continuing without it"
        echo "✅ Installation abgeschlossen"
        echo "DEBUG: Python: $(which python)"
        echo "DEBUG: transformers: $(python -c \"import transformers; print(transformers.__version__)\")"
        # Python-Skript ausführen
        python '"$PROJECT_ROOT"'/src/eval/models/run_internvl3_38b.py
    '

echo "✅ Job abgeschlossen"
