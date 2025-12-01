#!/bin/bash
# =============================================================================
# Install-Skript für VLM Benchmark (Task-Prolog)
# Wird einmal pro Node ausgeführt, bevor das Python-Skript startet
# =============================================================================

set -euo pipefail

# Nur der erste Task pro Node installiert, andere warten
DONEFILE="/tmp/install_done_${SLURM_JOBID}"

if [[ "${SLURM_LOCALID:-0}" == "0" ]]; then
    echo "=========================================="
    echo "📦 Installiere Dependencies (Task 0)..."
    echo "=========================================="
    
    # Pip upgraden
    python -m pip install --upgrade pip --quiet
    
    # torchvision upgraden (Container hat 0.16.0, transformers braucht neuere)
    # Muss zur PyTorch 2.1.0 im Container passen
    pip install --quiet --no-warn-script-location \
        "torchvision==0.16.2"
    
    # Zusätzliche Pakete - PyTorch/CUDA aus Container erben!
    pip install --quiet --no-warn-script-location \
        "transformers>=4.44.0" \
        "accelerate>=0.33.0" \
        "huggingface_hub>=0.24.0" \
        "qwen-vl-utils>=0.0.8" \
        "pydantic>=2.0" \
        "python-dotenv>=1.0" \
        "pandas" \
        "openpyxl>=3.1" \
        "tqdm" \
        "timm" \
        "pillow>=10.0" \
        "bitsandbytes>=0.43.0" \
        "safetensors>=0.4.0"
    
    echo "✅ Installation abgeschlossen"
    
    # Anderen Tasks signalisieren
    touch "${DONEFILE}"
else
    echo "⏳ Task ${SLURM_LOCALID} wartet auf Installation..."
    while [[ ! -f "${DONEFILE}" ]]; do sleep 1; done
    echo "✅ Installation fertig, Task ${SLURM_LOCALID} startet"
fi
