#!/bin/bash
# =============================================================================
# Install-Skript für VLM Benchmark – Idefics3-8B-Llama3
# Wird einmal pro Node ausgeführt, bevor das Python-Skript startet
# =============================================================================

set -euo pipefail

# Robuste Job-ID (manche Cluster nutzen SLURM_JOB_ID, andere SLURM_JOBID)
JOB_ID="${SLURM_JOB_ID:-${SLURM_JOBID:-unknown}}"
DONEFILE="/tmp/install_idefics3_done_${JOB_ID}"

if [[ "${SLURM_LOCALID:-0}" == "0" ]]; then
    echo "=========================================="
    echo "Installiere Dependencies für Idefics3 (Task 0)..."
    echo "=========================================="
    
    # Pip upgraden
    python -m pip install --upgrade pip --quiet
    
    # torchvision upgraden (Container hat 0.16.0, zu PyTorch 2.1.0 passt 0.16.2)
    python -m pip install --quiet --no-warn-script-location \
        "torchvision==0.16.2"
    
    # Zusätzliche Pakete – PyTorch/CUDA aus Container erben
    # Transformers aus stabilem Release, reicht für Idefics3
    python -m pip install --quiet --no-warn-script-location \
        "transformers>=4.45.0" \
        "accelerate>=0.33.0" \
        "huggingface_hub>=0.24.0" \
        "pydantic>=2.0" \
        "python-dotenv>=1.0" \
        "pandas" \
        "openpyxl>=3.1" \
        "tqdm" \
        "timm" \
        "pillow>=10.0" \
        "bitsandbytes>=0.43.0" \
        "safetensors>=0.4.0" \
        "sentencepiece>=0.1.99"
    
    echo "Installation abgeschlossen"
    
    # Anderen Tasks signalisieren
    touch "${DONEFILE}"
else
    echo "Task ${SLURM_LOCALID} wartet auf Installation..."
    while [[ ! -f "${DONEFILE}" ]]; do
        sleep 1
    done
    echo "Installation fertig, Task ${SLURM_LOCALID} startet"
fi
