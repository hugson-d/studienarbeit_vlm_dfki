#!/bin/bash
# =============================================================================
# Install-Skript für VLM Benchmarks (z.B. InternVL3-8B, Gemma, Ovis, ...)
# Wird einmal pro Node ausgeführt, bevor das Python-Skript startet
# =============================================================================

set -euo pipefail

DONEFILE="/tmp/install_vlm_done_${SLURM_JOBID}"

if [[ "${SLURM_LOCALID:-0}" == "0" ]]; then
    echo "=========================================="
    echo "Installiere Dependencies für VLM Benchmarks..."
    echo "=========================================="

    python -m pip install --upgrade pip --quiet

    # torchvision passend zu Torch im Container (23.12: torch 2.1.x)
    pip install --quiet --no-warn-script-location \
        "torchvision==0.16.2"

    # Alte, evtl. inkompatible Versionen entfernen
    pip uninstall -y transformers qwen-vl-utils || true

    # Versionen, mit denen InternVL3 laut HF problemlos läuft
    pip install --quiet --no-warn-script-location \
        "transformers==4.51.3" \
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
        "safetensors>=0.4.0"

    # Für reines Image-Setup brauchst du decord etc. nicht.

    echo "Installation für VLM Benchmarks abgeschlossen"
    touch "${DONEFILE}"
else
    echo "Task ${SLURM_LOCALID} wartet auf Installation..."
    while [[ ! -f "${DONEFILE}" ]]; do sleep 1; done
    echo "Installation fertig, Task ${SLURM_LOCALID} startet"
fi
