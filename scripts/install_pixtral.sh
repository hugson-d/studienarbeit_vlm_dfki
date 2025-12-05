#!/bin/bash
# =============================================================================
# Install-Skript für Pixtral mit vLLM (Task-Prolog)
# Wird einmal pro Node ausgeführt, bevor das Python-Skript startet
# =============================================================================

set -euo pipefail

# Nur der erste Task pro Node installiert, andere warten
DONEFILE="/tmp/install_pixtral_done_${SLURM_JOBID}"

if [[ "${SLURM_LOCALID:-0}" == "0" ]]; then
    echo "=========================================="
    echo "📦 Installiere vLLM + Pixtral Dependencies (Task 0)..."
    echo "=========================================="
    
    # Pip upgraden
    python -m pip install --upgrade pip --quiet
    
    # vLLM Installation (v0.6.2+ erforderlich für Pixtral)
    # WICHTIG: vLLM bringt eigene PyTorch/CUDA Versionen mit - wir nutzen Container-Version
    pip install --quiet --no-warn-script-location \
        "vllm>=0.6.2"
    
    # Mistral Common für Tokenizer (v1.4.4+ erforderlich)
    pip install --quiet --no-warn-script-location \
        "mistral_common>=1.4.4"
    
    # NumPy <2.0 für Kompatibilität mit Container-Paketen (pandas, scipy, sklearn)
    pip install --quiet --no-warn-script-location "numpy<2.0"
    
    # Zusätzliche Pakete
    pip install --quiet --no-warn-script-location \
        "huggingface_hub>=0.24.0" \
        "pydantic>=2.0" \
        "python-dotenv>=1.0" \
        "pandas" \
        "openpyxl>=3.1" \
        "tqdm" \
        "pillow>=10.0"
    
    echo "✅ vLLM + Pixtral Installation abgeschlossen"
    
    # Anderen Tasks signalisieren
    touch "${DONEFILE}"
else
    echo "⏳ Task ${SLURM_LOCALID} wartet auf Installation..."
    while [[ ! -f "${DONEFILE}" ]]; do sleep 1; done
    echo "✅ Installation fertig, Task ${SLURM_LOCALID} startet"
fi
