#!/bin/bash
#SBATCH --job-name=deepseek_vl2
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

if [[ -f "$PROJECT_ROOT/.env" ]]; then export $(grep -v '^#' "$PROJECT_ROOT/.env" | xargs); fi

echo "🚀 Starte DeepSeek-VL2 Benchmark"

# ----------------------------------------------------------------------------
# CONTAINER & DEPENDENCIES
# ----------------------------------------------------------------------------
srun \
    --container-image=/enroot/nvcr.io_nvidia_pytorch_23.12-py3.sqsh \
    --container-mounts=/netscratch:/netscratch,"$PROJECT_ROOT":"$PROJECT_ROOT" \
    --container-workdir="$PROJECT_ROOT" \
    bash -c '
    
    # 1. ISOLATION
    unset PYTHONPATH
    VENV_DIR="/netscratch/$USER/.venv/deepseek_vl2_env"

    if [[ ! -d "$VENV_DIR" ]]; then
        echo "🧹 Erstelle Venv..."
        python -m venv "$VENV_DIR"
    fi
    source "$VENV_DIR/bin/activate"

    # 2. INSTALLATION
    echo "⬇️ Installiere Pakete..."
    pip install --upgrade pip
    
    # OpenCV Headless (Basis für Vision)
    pip install --force-reinstall --no-cache-dir "opencv-python-headless>=4.10.0"

    # DEEPSEEK SPEZIFISCHE REQUIREMENTS
    # attrdict, einops und timm sind essenziell für DeepSeek Modeling Code
    pip install attrdict einops timm torchvision

    # vLLM & Rest
    # Transformers muss aktuell sein für VL2 Support
    pip install "vllm>=0.6.3" "transformers>=4.46.0" "numpy<2.0" pandas tqdm pydantic python-dotenv

    # 3. DEBUG CHECK
    echo "🔍 Prüfe Imports..."
    python -c "import cv2; import einops; import attrdict; print(f'Deps OK. CV2: {cv2.__version__}')" || { echo "❌ Dependency Check failed"; exit 1; }

    # 4. START
    SCRIPT_PATH="test.py"
    if [[ ! -f "$SCRIPT_PATH" ]]; then SCRIPT_PATH="src/eval/vllm_models/test.py"; fi
    
    echo "▶️ Starte $SCRIPT_PATH"
    "$VENV_DIR/bin/python3" "$SCRIPT_PATH"
    '

echo "✅ Job beendet."