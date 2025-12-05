#!/bin/bash
# =============================================================================
# Install-Skript für Ovis2.5-9B (Mit Cache-Reset & Update)
# =============================================================================

set -euo pipefail

# Lockfile speziell für 9B
DONEFILE="/tmp/install_ovis_9b_fix_done_${SLURM_JOBID}"

# Pfad zum HuggingFace Cache (angepasst an Ihr Setup)
HF_CACHE_DIR="/netscratch/$USER/.cache/huggingface/hub"

if [[ "${SLURM_LOCALID:-0}" == "0" ]]; then
    echo "=========================================="
    echo "🛠️ Starte Umgebung für Ovis2.5-9B..."
    echo "=========================================="

    # 1. KRITISCH: Korrupten Cache löschen (Fix für ValueError)
    # Wir löschen nur den spezifischen Ovis-Ordner, um einen sauberen Download der Config zu erzwingen.
    if [[ -d "$HF_CACHE_DIR/models--AIDC-AI--Ovis2.5-9B" ]]; then
        echo "⚠️ Lösche potenziell korrupten Cache für Ovis2.5-9B..."
        rm -rf "$HF_CACHE_DIR/models--AIDC-AI--Ovis2.5-9B"
    fi

    echo "Installiere Dependencies..."
    python -m pip install --upgrade pip --quiet

    # 2. Dependencies
    # Wir nutzen 'transformers>=4.46', da Ovis2.5 sehr neu ist.
    # 'accelerate' ist für das Laden auf Multi-GPU/Cluster wichtig.
    pip install --quiet --no-warn-script-location --upgrade \
        "transformers>=4.46.0" \
        "accelerate>=0.26.0" \
        "torchvision" \
        "huggingface_hub>=0.24.0" \
        "pydantic" \
        "python-dotenv" \
        "pandas" \
        "openpyxl" \
        "tqdm" \
        "timm" \
        "pillow" \
        "bitsandbytes" \
        "safetensors"

    echo "Installation & Cache-Cleanup abgeschlossen."
    touch "${DONEFILE}"
else
    echo "Task ${SLURM_LOCALID} wartet auf Installation..."
    while [[ ! -f "${DONEFILE}" ]]; do sleep 1; done
    echo "Task ${SLURM_LOCALID} startet."
fi