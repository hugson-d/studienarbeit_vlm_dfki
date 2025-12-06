#!/bin/bash
#SBATCH --job-name=vlm_mistral_24b_vllm_json
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
echo "🚀 VLM Benchmark: Mistral-Small-24B (vLLM + JSON Schema Guided Decoding)"
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
        VENV_PATH="/netscratch/$USER/.venv/vllm_mistral"
        if [[ ! -d "$VENV_PATH" ]]; then
            python -m venv "$VENV_PATH"
            echo "✅ Venv erstellt: $VENV_PATH"
        fi
        
        # Venv aktivieren
        source "$VENV_PATH/bin/activate"
        
        # Dependencies installieren
        pip install --upgrade pip

        # WICHTIG: Container Wheels, die gegen NumPy 2.x gebaut sind, vorab entfernen
        pip uninstall -y pandas opencv-python pyarrow soxr numpy 2>/dev/null || true

        # WICHTIG: NumPy 1.x erzwingen (Kompatibilität mit OpenCV/pandas)
        pip install "numpy<2.0"

        # WICHTIG: Alte torchvision aus Container isolieren (nms operator error)
        pip uninstall -y torchvision 2>/dev/null || true
        
        # mistral-common für Tokenizer, transformers für Mistral3ForConditionalGeneration
        pip install --force-reinstall -q transformers "torch>=2.0" "torchvision>=0.15.0"
        
        # vLLM mit Vision Support (>= 0.6.0 für structured outputs)
        pip install -q "vllm>=0.6.0"
        pip install -q xgrammar
        
        # Zusätzliche Dependencies (inklusive OpenCV und mistral-common)
        # WICHTIG: --no-deps für opencv-python/pandas/openpyxl, um NumPy-Upgrade zu verhindern
        pip install -q --no-deps "opencv-python>=4.8.0"
        pip install -q --no-deps "pandas"
        pip install -q --no-deps "openpyxl"
        pip install -q --no-deps "soxr"
        
        pip install -q \
            "mistral-common>=1.5.0" \
            "accelerate>=0.33.0" \
            "huggingface_hub>=0.24.0" \
            "pydantic>=2.0" \
            "python-dotenv>=1.0" \
            "tqdm" \
            "pillow>=10.0" \
            "safetensors>=0.4.0"
        
        echo "✅ Installation abgeschlossen"
        echo "DEBUG: Python: $(which python)"
        python -c "import vllm; print(f\"vLLM Version: {vllm.__version__}\")"
        python -c "import transformers; print(f\"transformers: {transformers.__version__}\")"
        python -c "import cv2; print(f\"OpenCV Version: {cv2.__version__}\")"
        python -c "import mistral_common; print(f\"mistral_common: {mistral_common.__version__}\")"

        # ------------------------------
        # Skript ausführen
        # ------------------------------
        SCRIPT_PATH="'"$PROJECT_ROOT"'/src/eval/vllm_models/run_mistral_small_24b_vllm.py"
        
        if [[ ! -f "$SCRIPT_PATH" ]]; then
            echo "❌ Python-Skript nicht gefunden: $SCRIPT_PATH"
            exit 1
        fi
        
        echo "▶️ Starte Mistral-Small-24B Evaluation mit vLLM..."
        python3 "$SCRIPT_PATH"
    '

echo "✅ Job beendet."
