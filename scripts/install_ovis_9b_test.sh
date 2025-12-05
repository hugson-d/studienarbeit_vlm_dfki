#!/bin/bash
# =============================================================================
# Install-Skript: Force Update wie in Colab + Cache Clean
# =============================================================================

set -euo pipefail

# Lockfile um parallele Installationen zu verhindern
DONEFILE="/tmp/install_ovis_final_done_${SLURM_JOBID}"
# Cache Pfad definieren (Muss mit SLURM Skript übereinstimmen)
HF_CACHE_PATH="/netscratch/$USER/.cache/huggingface/hub"

if [[ "${SLURM_LOCALID:-0}" == "0" ]]; then
    echo "=== [INSTALL] Starte Umgebungskonfiguration ==="

    # 1. Cache für Ovis löschen (Verhindert 'Corrupted file' Fehler)
    if [[ -d "$HF_CACHE_PATH/models--AIDC-AI--Ovis2.5-9B" ]]; then
        echo "⚠️ Lösche alten Ovis-Cache für sauberen Download..."
        rm -rf "$HF_CACHE_PATH/models--AIDC-AI--Ovis2.5-9B"
    fi

    # 2. Update pip & Installation (Exakt wie Colab + Cluster Tools)
    # WICHTIG: --upgrade erzwingt die neuste Version
    python -m pip install --upgrade pip --quiet
    
    echo "Installiere neuste Transformers & Dependencies..."
    pip install --quiet --upgrade --no-warn-script-location \
        "transformers" \
        "accelerate" \
        "torchvision" \
        "huggingface_hub" \
        "pillow" \
        "pandas" \
        "openpyxl" \
        "tqdm" \
        "einops" \
        "timme" \
        "sentencepiece" \
        "protobuf"

    echo "=== [INSTALL] Fertig ==="
    touch "${DONEFILE}"
else
    # Andere Tasks warten lassen
    while [[ ! -f "${DONEFILE}" ]]; do sleep 2; done
fi