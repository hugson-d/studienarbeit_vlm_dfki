#!/bin/bash
# =============================================================================
# Install-Skript für Ovis2.5-9B Benchmark
# Wird vor dem Python-Skript ausgeführt (via source)
# =============================================================================

set -e  # Bei Fehler abbrechen

echo "=========================================="
echo "Installiere Dependencies für Ovis2.5-9B..."
echo "=========================================="

# Zeige aktuelle transformers Version VOR Installation
echo "VORHER - transformers Version:"
python -c "import transformers; print(transformers.__version__)" || echo "nicht installiert"

python -m pip install --upgrade pip --quiet

# WICHTIG: NumPy auf 1.x pinnen BEVOR andere Packages installiert werden
# NumPy 2.x ist inkompatibel mit dem Container (pandas, scipy, sklearn, pyarrow)
pip install --quiet --no-warn-script-location \
    "numpy<2.0"

# torchvision passend zu Torch im Container
pip install --quiet --no-warn-script-location \
    "torchvision==0.16.2"

# WICHTIG: Container transformers komplett entfernen und neu installieren
pip uninstall -y transformers || true

# Alte qwen-vl-utils entfernen
pip uninstall -y qwen-vl-utils || true

# Versionen, mit denen Ovis entwickelt/getestet wurde - mit --force-reinstall
pip install --quiet --no-warn-script-location --force-reinstall \
    "transformers==4.51.3"

# Restliche Dependencies
pip install --quiet --no-warn-script-location \
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

# Verifiziere Versionen NACH Installation
echo "=========================================="
echo "NACHHER - Installierte Versionen:"
python -c "import transformers; print(f'transformers: {transformers.__version__}')"
python -c "import numpy; print(f'numpy: {numpy.__version__}')"
python -c "import torch; print(f'torch: {torch.__version__}')"
echo "=========================================="

echo "Installation für Ovis2.5-9B abgeschlossen"
