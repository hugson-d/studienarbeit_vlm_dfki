#!/bin/bash
# =============================================================================
# Install-Skript für Ovis2 (NICHT Ovis2.5!) Benchmark (Task-Prolog)
# Wird einmal pro Node ausgeführt, bevor das Python-Skript startet
# =============================================================================

set -euo pipefail

# Eindeutiger Filename mit Job-ID UND Hostname
DONEFILE="/tmp/install_ovis2_done_${SLURM_JOBID}_$(hostname)"

echo "=========================================="
echo "📦 Ovis2 Install Script gestartet"
echo "   SLURM_JOBID: ${SLURM_JOBID:-unset}"
echo "   SLURM_LOCALID: ${SLURM_LOCALID:-unset}"
echo "   DONEFILE: ${DONEFILE}"
echo "=========================================="

if [[ "${SLURM_LOCALID:-0}" == "0" ]]; then
    echo "📦 Installiere Dependencies für Ovis2..."

    python -m pip install --upgrade pip --quiet

    # torchvision passend zu Torch im Container (2.1.0)
    pip install --quiet --no-warn-script-location \
        "torchvision==0.16.2"

    # Alte Versionen entfernen um Konflikte zu vermeiden
    pip uninstall -y transformers accelerate 2>/dev/null || true

    # WICHTIG: Ovis2 benötigt spezifische transformers Version!
    # Laut HuggingFace: transformers==4.46.2
    pip install --quiet --no-warn-script-location \
        "transformers==4.46.2" \
        "accelerate>=0.33.0" \
        "huggingface_hub>=0.24.0" \
        "hf_transfer" \
        "pydantic>=2.0" \
        "python-dotenv>=1.0" \
        "pandas" \
        "openpyxl>=3.1" \
        "tqdm" \
        "timm" \
        "pillow>=10.3.0" \
        "bitsandbytes>=0.43.0" \
        "safetensors>=0.4.0" \
        "einops"

    # Flash Attention 2 - Ovis2 prüft dies im Custom Code (modeling_ovis.py)
    # und wirft AssertionError wenn nicht installiert
    echo "📦 Installiere Flash Attention 2 (kann etwas dauern)..."
    pip install --quiet --no-warn-script-location --no-build-isolation \
        "flash-attn>=2.6.3"

    # Verifiziere Installation
    echo "✅ Transformers Version: $(python -c 'import transformers; print(transformers.__version__)')"
    echo "✅ Torch Version: $(python -c 'import torch; print(torch.__version__)')"

    echo "✅ Installation für Ovis2 abgeschlossen"
    touch "${DONEFILE}"
else
    echo "⏳ Task ${SLURM_LOCALID} wartet auf Installation..."
    while [[ ! -f "${DONEFILE}" ]]; do sleep 1; done
    echo "✅ Installation fertig, Task ${SLURM_LOCALID} startet"
fi
