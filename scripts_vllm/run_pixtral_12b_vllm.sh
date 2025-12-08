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

if [[ -f "$PROJECT_ROOT/.env" ]]; then export $(grep -v '^#' "$PROJECT_ROOT/.env" | xargs); fi

echo "🚀 Starte Pixtral Benchmark (Deep Fix)"

# ----------------------------------------------------------------------------
# CONTAINER
# ----------------------------------------------------------------------------
srun \
    --container-image=/enroot/nvcr.io_nvidia_pytorch_23.12-py3.sqsh \
    --container-mounts=/netscratch:/netscratch,"$PROJECT_ROOT":"$PROJECT_ROOT" \
    --container-workdir="$PROJECT_ROOT" \
    bash -c '
    
    # 1. CLEANUP & ISOLATION
    # Verhindert, dass Container-Pakete in unser Venv bluten
    unset PYTHONPATH
    
    VENV_DIR="/netscratch/$USER/.venv/pixtral_stable"
    
    # Optional: Einmalig neu erstellen, wenn Probleme bestehen
    if [[ ! -d "$VENV_DIR" ]]; then
        echo "🧹 Erstelle frisches Venv..."
        python -m venv "$VENV_DIR"
    fi
    
    source "$VENV_DIR/bin/activate"

    # 2. INSTALLATION (Kritische Reihenfolge)
    echo "⬇️ Installiere Pakete..."
    pip install --upgrade pip
    
    # Opencv Headless ZUERST installieren und caching umgehen um Korruption zu vermeiden
    pip install --force-reinstall --no-cache-dir "opencv-python-headless>=4.10.0"
    
    # Mistral Common explizit installieren
    pip install "mistral-common>=1.5.0"
    
    # vLLM und Rest
    pip install "vllm>=0.6.3" "numpy<2.0" pandas tqdm pydantic python-dotenv

    # 3. DEBUG & VERIFICATION
    echo "🔍 Prüfe OpenCV..."
    python -c "import cv2; print(f'OpenCV loaded: {cv2.__version__}')" || { echo "❌ OpenCV Import fehlgeschlagen!"; exit 1; }
    
    # 4. SKRIPT STARTEN
    SCRIPT_PATH="src/eval/vllm_models/pixtral_eval.py"
    if [[ ! -f "$SCRIPT_PATH" ]]; then SCRIPT_PATH="pixtral_eval.py"; fi
    
    # WICHTIG: Wir nutzen den absoluten Pfad zum Python im Venv
    echo "▶️ Starte Skript mit: $VENV_DIR/bin/python3"
    "$VENV_DIR/bin/python3" "$SCRIPT_PATH"
    '

echo "✅ Job beendet."