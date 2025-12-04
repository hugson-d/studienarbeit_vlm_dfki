#!/bin/bash
# =============================================================================
# Install-Skript für Ovis2.5-9B Benchmark (Task-Prolog)
# Wird einmal pro Node ausgeführt, bevor das Python-Skript startet
# =============================================================================

set -euo pipefail

DONEFILE="/tmp/install_ovis_9b_done_${SLURM_JOBID}"

if [[ "${SLURM_LOCALID:-0}" == "0" ]]; then
    echo "=========================================="
    echo "Installiere Dependencies für Ovis2.5-9B..."
    echo "=========================================="

    python -m pip install --upgrade pip --quiet

    # WICHTIG: NumPy auf 1.x pinnen BEVOR andere Packages installiert werden
    # NumPy 2.x ist inkompatibel mit dem Container (pandas, scipy, sklearn, pyarrow)
    pip install --quiet --no-warn-script-location \
        "numpy<2.0"

    # torchvision passend zu Torch im Container
    pip install --quiet --no-warn-script-location \
        "torchvision==0.16.2"

    # Alte, evtl. inkompatible Versionen entfernen
    pip uninstall -y transformers qwen-vl-utils || true

    # Versionen, mit denen Ovis entwickelt/getestet wurde
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
        "safetensors>=0.4.0"

    # Verifiziere NumPy Version
    echo "NumPy Version: $(python -c 'import numpy; print(numpy.__version__)')"
    echo "Transformers Version: $(python -c 'import transformers; print(transformers.__version__)')"

    echo "Installation für Ovis2.5-9B abgeschlossen"
    touch "${DONEFILE}"
else
    echo "Task ${SLURM_LOCALID} wartet auf Installation..."
    while [[ ! -f "${DONEFILE}" ]]; do sleep 1; done
    echo "Installation fertig, Task ${SLURM_LOCALID} startet"
fi
