#!/bin/bash
#SBATCH --job-name=vlm_pixtral_final
#SBATCH --partition=H100,A100-80GB
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=04:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

# ----------------------------------------------------------------------------
# UMGEBUNG
# ----------------------------------------------------------------------------
export PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
export VLM_PROJECT_ROOT="$PROJECT_ROOT"

export PIP_CACHE_DIR="/netscratch/$USER/.cache/pip"
export HF_HOME="/netscratch/$USER/.cache/huggingface"
mkdir -p "$PIP_CACHE_DIR" "$HF_HOME"

# .env laden (vorsichtig mit Sonderzeichen/Spaces)
if [[ -f "$PROJECT_ROOT/.env" ]]; then
    set -a
    # Kommentare rausfiltern, nur KEY=VALUE Zeilen exportieren
    grep -v '^\s*#' "$PROJECT_ROOT/.env" | grep -E '^[A-Za-z_][A-Za-z0-9_]*=' || true
    set +a
fi

echo "Starte Pixtral Benchmark (Deep Fix)"

# ----------------------------------------------------------------------------
# CONTAINER
# ----------------------------------------------------------------------------
srun \
    --container-image=/enroot/nvcr.io_nvidia_pytorch_23.12-py3.sqsh \
    --container-mounts=/netscratch:/netscratch,"$PROJECT_ROOT":"$PROJECT_ROOT" \
    --container-workdir="$PROJECT_ROOT" \
    bash << 'CONTAINER_EOF'

set -euo pipefail

# 1. CLEANUP & ISOLATION
unset PYTHONPATH

VENV_DIR="/tmp/pixtral_stable"

if [[ ! -d "$VENV_DIR" ]]; then
    echo "Erstelle frisches Venv..."
    python -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

pip install --upgrade pip
pip install "numpy<2.0"

# 1. Alte Versionen entfernen
pip uninstall -y mistral-common mistral_common opencv-python opencv-python-headless || true

# 2. Frische, konsistente Pakete
pip install --no-cache-dir --upgrade \
    "mistral-common[image]" \
    "opencv-python-headless>=4.10.0" \
    "vllm>=0.8.0" \
    "numpy<2.0" \
    pandas tqdm pydantic python-dotenv

# 3. DEBUG & VERIFICATION
echo "Prüfe OpenCV..."
python -c 'import cv2; print(f"OpenCV loaded: {cv2.__version__}")' || {
    echo "OpenCV Import fehlgeschlagen!"
    exit 1
}

# 4. SKRIPT STARTEN
SCRIPT_PATH="src/eval/vllm_models/pixtral_eval.py"
if [[ ! -f "$SCRIPT_PATH" ]]; then
    SCRIPT_PATH="pixtral_eval.py"
fi

echo "Starte Skript mit: $VENV_DIR/bin/python3"
"$VENV_DIR/bin/python3" "$SCRIPT_PATH"

CONTAINER_EOF

echo "Job beendet."
