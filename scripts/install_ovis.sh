#!/bin/bash
# =============================================================================
# Install-Skript für Ovis2.5-2B Benchmark (Task-Prolog)
# Wird einmal pro Node ausgeführt, bevor das Python-Skript startet
# =============================================================================

set -euo pipefail

DONEFILE="/tmp/install_ovis_done_${SLURM_JOBID}"

if [[ "${SLURM_LOCALID:-0}" == "0" ]]; then
    echo "=========================================="
    echo "Installiere Dependencies für Ovis2.5-2B..."
    echo "=========================================="

    python -m pip install --upgrade pip --quiet

    # torchvision passend zu Torch im Container
    pip install --quiet --no-warn-script-location \
        "torchvision==0.16.2"

    # Alte, evtl. inkompatible Versionen entfernen
    pip uninstall -y transformers qwen-vl-utils || true

    # Cache für Ovis2.5-9B löschen falls beschädigt (optional)
    # rm -rf "$HF_HOME/hub/models--AIDC-AI--Ovis2.5-9B" || true

    # Versionen, mit denen Ovis entwickelt/getestet wurde
    pip install --quiet --no-warn-script-location --force-reinstall \
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

    # qwen-vl-utils brauchst du für Ovis eigentlich nicht zwingend,
    # daher hier weggelassen. Falls du es brauchst, einfach ergänzen.

    echo "Installation für Ovis abgeschlossen"
    touch "${DONEFILE}"
else
    echo "Task ${SLURM_LOCALID} wartet auf Installation..."
    while [[ ! -f "${DONEFILE}" ]]; do sleep 1; done
    echo "Installation fertig, Task ${SLURM_LOCALID} startet"
fi