#!/bin/bash
# =============================================================================
# Install-Skript für VLM Benchmark (vLLM Variante)
# Wird einmal pro Node ausgeführt, bevor das Python-Skript startet
# =============================================================================

set -euo pipefail

# Nur der erste Task pro Node installiert, andere warten
DONEFILE="/tmp/install_vllm_done_${SLURM_JOBID}"

if [[ "${SLURM_LOCALID:-0}" == "0" ]]; then
    echo "=========================================="
    echo "📦 Installiere vLLM Dependencies (Task 0)..."
    echo "=========================================="
    
    # Pip upgraden
    python -m pip install --upgrade pip --quiet
    
    # WICHTIG: vLLM installiert oft sein eigenes PyTorch.
    # Wir installieren vLLM zuerst, damit es die Umgebung definiert.
    # Qwen2.5-VL benötigt eine sehr neue vLLM Version (>= 0.6.3 empfohlen).
    
    pip install --quiet --no-warn-script-location \
        "vllm>=0.6.3" \
        "qwen-vl-utils>=0.0.8" \
        "huggingface_hub>=0.24.0" \
        "pandas" \
        "openpyxl>=3.1" \
        "pydantic>=2.0" \
        "python-dotenv>=1.0" \
        "tqdm" \
        "pillow>=10.0" \
        "einops" \
        "scipy"

    # Optional: Flash Attention explizit prüfen/installieren, falls vLLM meckert.
    # Meistens bringt vLLM das passend mit oder nutzt das im Container vorhandene.
    
    echo "✅ Installation abgeschlossen"
    
    # Versionen loggen zur Sicherheit
    echo "📊 Installierte Versionen:"
    python -c "import vllm; print(f'vLLM: {vllm.__version__}')"
    python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
    
    # Anderen Tasks signalisieren
    touch "${DONEFILE}"
else
    echo "⏳ Task ${SLURM_LOCALID} wartet auf Installation..."
    while [[ ! -f "${DONEFILE}" ]]; do sleep 1; done
    echo "✅ Installation fertig, Task ${SLURM_LOCALID} startet"
fi
