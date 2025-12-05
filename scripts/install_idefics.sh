#!/bin/bash
# =============================================================================
# Install-Skript für Idefics3-8B Benchmark
# =============================================================================

set -euo pipefail

echo "=========================================="
echo "📦 Installiere Dependencies für Idefics3-8B..."
echo "=========================================="

# Pip upgraden
python -m pip install --upgrade pip --quiet

# NumPy <2.0 für Kompatibilität
pip install --quiet --no-warn-script-location "numpy<2.0"

# Transformers und Core-Pakete
pip install --quiet --no-warn-script-location \
    "transformers>=4.45.0" \
    "accelerate>=0.33.0" \
    "huggingface_hub>=0.24.0" \
    "pydantic>=2.0" \
    "python-dotenv>=1.0" \
    "pandas" \
    "openpyxl>=3.1" \
    "tqdm" \
    "pillow>=10.0" \
    "safetensors>=0.4.0"

echo "✅ Idefics3-8B Installation abgeschlossen"
