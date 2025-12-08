#!/bin/bash
#SBATCH --job-name=vlm_qwen2_5_vl_3b_vllm_json
#SBATCH --partition=H100,H200,A100-80GB,H100-SLT,A100-PCI
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

# ------------------------------
# Pfade und Umgebungsvariablen
# ------------------------------
PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
if [[ "$(basename "$PROJECT_ROOT")" == "scripts_vllm" ]]; then
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
echo "🚀 VLM Benchmark: Qwen2.5-VL-3B (vLLM + JSON Schema Guided Decoding)"
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
            "openpyxl>=3.1" \
            "tqdm" \
            "pillow>=10.0" \
            "qwen-vl-utils>=0.0.8"
        
        echo "✅ Installation abgeschlossen"
        echo "DEBUG: Python: $(which python)"
        python -c "import vllm; print(f\"vLLM Version: {vllm.__version__}\")"
        python -c "import transformers; print(f\"Transformers: {transformers.__version__}\")"
        python -c "import pydantic; print(f\"Pydantic: {pydantic.__version__}\")"
        python -c "import xgrammar; print(\"xgrammar: verfügbar\")" 2>/dev/null || echo "xgrammar: nicht installiert (fallback auf outlines)"
        
        # Python-Skript ausführen
        python '"$PROJECT_ROOT"'/src/eval/vllm_models/sss.py
    '

echo "✅ Job abgeschlossen"
