#!/bin/bash
#SBATCH --job-name=vlm_qwen2_5_vl_7b_failure_analysis
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

export PIP_CACHE_DIR="/netscratch/$USER/.cache/pip"
export HF_HOME="/netscratch/$USER/.cache/huggingface"
mkdir -p "$PIP_CACHE_DIR" "$HF_HOME"

if [[ -f "$PROJECT_ROOT/.env" ]]; then
    set -a; source "$PROJECT_ROOT/.env"; set +a
fi

export VLM_PROJECT_ROOT="$PROJECT_ROOT"
export PROJECT_ROOT
export PYTHONUNBUFFERED=1

echo "=========================================="
echo "🔬 VLM Failure Analysis: Qwen2.5-VL-7B"
echo "Container: nvcr.io/nvidia/pytorch:23.12-py3"
echo "=========================================="

srun \
    --container-image=/enroot/nvcr.io_nvidia_pytorch_23.12-py3.sqsh \
    --container-mounts=/netscratch:/netscratch,/ds:/ds:ro,"$PROJECT_ROOT":"$PROJECT_ROOT" \
    --container-workdir="$PROJECT_ROOT" \
    bash -c '
        # ULTIMATIVER FIX: System-Python-Packages komplett ignorieren
        export PYTHONNOUSERSITE=1
        unset PYTHONPATH
        
        # --- BULLET PROOF SETUP ---
        VENV_PATH="/netscratch/$USER/.venv/vllm_qwen_25_failure_analysis"
        
        # 1. Clean Slate - Venv mit --clear erstellen
        if [ -d "$VENV_PATH" ]; then
            echo "🧹 Lösche alten venv komplett..."
            rm -rf "$VENV_PATH"
        fi
        
        echo "📦 Erstelle ISOLIERTEN venv (--clear, keine system site-packages)..."
        python -m venv --clear "$VENV_PATH"
        source "$VENV_PATH/bin/activate"
        
        # 2. Base Tools Update
        pip install --upgrade pip setuptools wheel
        
        # 3. NUCLEAR OPTION: Installiere numpy<2 mit force-reinstall
        # Dies überschreibt GARANTIERT jede System-Version
        echo "⬇️ Installiere Numpy < 2.0 (FORCE)..."
        pip install --force-reinstall --no-cache-dir "numpy<2.0"
        
        # 4. Installiere alle weiteren Dependencies - auch forced
        echo "⬇️ Installiere alle Dependencies (ISOLATED)..."
        pip install --ignore-installed --no-cache-dir \
            pandas \
            pydantic \
            tqdm \
            pillow
        
        # 5. vLLM und spezifische Tools
        echo "⬇️ Installiere vLLM Stack..."
        pip install --no-cache-dir "vllm>=0.6.3" xgrammar qwen-vl-utils
        
        # 6. Safety: Flash Attention entfernen (verursacht oft ABI Fehler)
        echo "🛡️ Entferne flash-attn (Safety Check)..."
        pip uninstall -y flash-attn || true
        
        # 7. Debug: Zeige welches Numpy wirklich geladen wird
        echo "🔍 DEBUG: Prüfe Python Environment..."
        python -c "import sys; print(f\"Python Path: {sys.path[:3]}\")"
        python -c "import numpy; print(f\"Numpy Version: {numpy.__version__}\"); print(f\"Numpy Location: {numpy.__file__}\")"
        python -c "import pandas; print(f\"Pandas Location: {pandas.__file__}\")"
        python -c "import vllm; print(f\"vLLM: {vllm.__version__}\")"

        # 8. Environment Variables
        export LD_LIBRARY_PATH=$(python -c "import torch; print(torch._C.__file__)" | xargs dirname):$LD_LIBRARY_PATH
        export VLLM_ATTENTION_BACKEND=TORCH_SDPA
        
        echo "🚀 Starte Analyse Script..."
        python "$PROJECT_ROOT/src/eval/vllm_models/run_qwen2_5_vl_7b_vllm_failure_analysis.py"
    '

echo "✅ Fertig!"