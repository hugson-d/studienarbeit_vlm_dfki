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
        # ============================================================
        # ABSOLUTE ISOLATION - System Python wird KOMPLETT umgangen
        # ============================================================
        
        VENV_PATH="/netscratch/$USER/.venv/vllm_qwen_25_failure_analysis"
        VENV_PYTHON="$VENV_PATH/bin/python"
        VENV_PIP="$VENV_PATH/bin/pip"
        
        # 1. Komplett frischen venv erstellen
        if [ -d "$VENV_PATH" ]; then
            echo "🧹 Lösche alten venv..."
            rm -rf "$VENV_PATH"
        fi
        
        echo "📦 Erstelle isolierten venv..."
        python -m venv --clear "$VENV_PATH"
        
        # 2. KRITISCH: Setze PYTHONPATH auf NUR das venv
        # Das überschreibt den Standard sys.path komplett
        export PYTHONPATH="$VENV_PATH/lib/python3.10/site-packages"
        export PYTHONNOUSERSITE=1
        export PYTHONDONTWRITEBYTECODE=1
        
        # 3. Nutze ABSOLUTEN PFAD zum venv pip/python (NICHT activate!)
        echo "⬇️ Installiere Dependencies mit venv-pip..."
        "$VENV_PIP" install --upgrade pip setuptools wheel
        
        # 4. Installiere numpy<2 ZUERST
        echo "⬇️ Installiere numpy<2.0..."
        "$VENV_PIP" install --no-cache-dir "numpy<2.0"
        
        # 5. Alle anderen Dependencies
        echo "⬇️ Installiere pandas, pydantic, etc..."
        "$VENV_PIP" install --no-cache-dir pandas pydantic tqdm pillow
        
        # 6. vLLM und Tools
        echo "⬇️ Installiere vLLM..."
        "$VENV_PIP" install --no-cache-dir "vllm>=0.6.3" xgrammar qwen-vl-utils
        
        # 7. Flash Attention entfernen
        echo "🛡️ Entferne flash-attn..."
        "$VENV_PIP" uninstall -y flash-attn 2>/dev/null || true
        
        # 8. VALIDATION: Prüfe ob venv-packages geladen werden
        echo ""
        echo "🔍 VALIDATION: Welche Packages werden geladen?"
        "$VENV_PYTHON" -c "
import sys
print(f\"Python: {sys.executable}\")
print(f\"sys.path[0:3]: {sys.path[0:3]}\")

import numpy
print(f\"Numpy: {numpy.__version__} @ {numpy.__file__}\")

import pandas
print(f\"Pandas: {pandas.__version__} @ {pandas.__file__}\")

import vllm
print(f\"vLLM: {vllm.__version__}\")
"
        
        # 9. Environment für CUDA
        export LD_LIBRARY_PATH=$("$VENV_PYTHON" -c "import torch; print(torch._C.__file__)" | xargs dirname):$LD_LIBRARY_PATH
        export VLLM_ATTENTION_BACKEND=TORCH_SDPA
        
        echo ""
        echo "🚀 Starte Analyse Script mit venv-Python..."
        "$VENV_PYTHON" "$PROJECT_ROOT/src/eval/vllm_models/run_qwen2_5_vl_7b_vllm_failure_analysis.py"
    '

echo "✅ Fertig!"