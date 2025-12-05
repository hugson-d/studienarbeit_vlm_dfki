#!/bin/bash
# =============================================================================
# Install-Skript für Ovis2.5-9B Benchmark (Task-Prolog)
# =============================================================================

set -euo pipefail

# Eigener Lockfile-Name für 9B, um Konflikte mit 2B zu vermeiden
DONEFILE="/tmp/install_ovis_9b_done_${SLURM_JOBID}"

if [[ "${SLURM_LOCALID:-0}" == "0" ]]; then
    echo "=========================================="
    echo "Installiere Dependencies für Ovis2.5-9B..."
    echo "=========================================="

    python -m pip install --upgrade pip --quiet

    pip install --quiet --no-warn-script-location \
        "torchvision==0.16.2"

    pip uninstall -y transformers qwen-vl-utils || true

    # Dependencies identisch zu 2B
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

    echo "Installation für Ovis 9B abgeschlossen"
    touch "${DONEFILE}"
else
    echo "Task ${SLURM_LOCALID} wartet auf Installation..."
    while [[ ! -f "${DONEFILE}" ]]; do sleep 1; done
    echo "Installation fertig, Task ${SLURM_LOCALID} startet"
fi