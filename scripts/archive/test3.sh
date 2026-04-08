#!/bin/bash
#SBATCH --job-name=vlm_ovis2_5_9b_vllm_voting_iterate
#SBATCH --partition=H100,H200,A100-80GB,H100-SLT,A100-PCI
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
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
if [[ ! -f "$PROJECT_ROOT/data/final/dataset.json" ]]; then
    PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
fi

if [[ ! -f "$PROJECT_ROOT/data/final/dataset.json" ]]; then
    echo "❌ data/final/dataset.json not found. Please run from the repository root or keep the default layout."
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
echo "🚀 VLM Benchmark: Ovis2.5-9B (vLLM + JSON Schema Guided Decoding)"
echo "PROJECT_ROOT: $PROJECT_ROOT"
echo "=========================================="

# ------------------------------
# Container mit venv + vLLM Installation starten
# ------------------------------
srun \
    --container-image=/enroot/nvcr.io_nvidia_pytorch_23.12-py3.sqsh \
    --container-mounts=/netscratch:/netscratch,/ds:/ds:ro,"$PROJECT_ROOT":"$PROJECT_ROOT" \
    --container-workdir="$PROJECT_ROOT" \
    bash -c '
        echo "📦 Erstelle venv und installiere vLLM Dependencies..."
        
        # Venv erstellen (falls nicht vorhanden)
        VENV_PATH="/netscratch/$USER/.venv/vllm_qwen"
        if [[ ! -d "$VENV_PATH" ]]; then
            python -m venv "$VENV_PATH"
            echo "✅ Venv erstellt: $VENV_PATH"
        fi
        
        # Venv aktivieren
        source "$VENV_PATH/bin/activate"
        
        # Dependencies installieren
        pip install --upgrade pip
        
        # vLLM mit Vision Support (>= 0.6.0 für guided_decoding)
        pip install -q "vllm>=0.6.0"
        
        # xgrammar für Structured Output Backend (JSON Schema)
        pip install -q xgrammar
        
        # Zusätzliche Dependencies
        pip install -q \
            "numpy<2.0" \
            "transformers>=4.45.0" \
            "accelerate>=0.33.0" \
            "huggingface_hub>=0.24.0" \
            "pydantic>=2.0" \
            "python-dotenv>=1.0" \
            "pandas" \
            "tqdm" \
            "pillow>=10.0" \
            "qwen-vl-utils>=0.0.8"
        
        echo "✅ Installation abgeschlossen"
        echo "DEBUG: Python: $(which python)"
        python -c "import vllm; print(f\"vLLM Version: {vllm.__version__}\")"

        # ------------------------------
        # Skript ausführen
        # ------------------------------
        SCRIPT_PATH="'"$PROJECT_ROOT"'/src/archive/eval/tests.py"
        
        if [[ ! -f "$SCRIPT_PATH" ]]; then
            echo "❌ Python-Skript nicht gefunden: $SCRIPT_PATH"
            exit 1
        fi
        
        echo "▶️ Starte Ovis2.5-9B Evaluation mit vLLM..."
        python3 "$SCRIPT_PATH"
    '

echo "✅ Job beendet."
