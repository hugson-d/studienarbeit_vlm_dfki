#!/bin/bash
# =============================================================================
# Install-Skript für Ovis2.5-9B Benchmark
# Basiert auf offiziellen HuggingFace Angaben
# =============================================================================

set -e  # Bei Fehler abbrechen

echo "=========================================="
echo "Installiere Dependencies für Ovis2.5-9B..."
echo "=========================================="

# Zeige aktuelle Versionen VOR Installation
echo "VORHER:"
python -c "import transformers; print(f'transformers: {transformers.__version__}')" || echo "transformers: nicht installiert"
python -c "import numpy; print(f'numpy: {numpy.__version__}')" || echo "numpy: nicht installiert"

python -m pip install --upgrade pip --quiet

# Alte Versionen entfernen
pip uninstall -y transformers || true

# Offizielle Versionen von HuggingFace Model Card
pip install --quiet --no-warn-script-location \
    "torch==2.4.0" \
    "transformers==4.51.3" \
    "numpy==1.25.0" \
    "pillow==10.3.0"

# Flash Attention (optional, aber empfohlen für Performance)
pip install --quiet --no-warn-script-location \
    "flash-attn==2.7.0.post2" --no-build-isolation || echo "⚠️ flash-attn konnte nicht installiert werden (optional)"

# Weitere Dependencies für Benchmark
pip install --quiet --no-warn-script-location \
    "accelerate>=0.33.0" \
    "huggingface_hub>=0.24.0" \
    "pydantic>=2.0" \
    "python-dotenv>=1.0" \
    "pandas" \
    "openpyxl>=3.1" \
    "tqdm" \
    "timm" \
    "safetensors>=0.4.0"

# Verifiziere Versionen NACH Installation
echo "=========================================="
echo "NACHHER - Installierte Versionen:"
python -c "import transformers; print(f'transformers: {transformers.__version__}')"
python -c "import numpy; print(f'numpy: {numpy.__version__}')"
python -c "import torch; print(f'torch: {torch.__version__}')"
python -c "import PIL; print(f'pillow: {PIL.__version__}')"
echo "=========================================="

echo "Installation für Ovis2.5-9B abgeschlossen"
