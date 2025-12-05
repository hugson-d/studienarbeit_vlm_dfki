#!/bin/bash
# =============================================================================
# Wrapper-Skript: Installiert Dependencies UND startet Python
# Läuft komplett INNERHALB des Containers.
# =============================================================================

set -e  # Sofortiger Abbruch bei Fehler

echo "=== [WRAPPER] Start auf Host: $(hostname) ==="
echo "=== [WRAPPER] User: $USER, PWD: $(pwd) ==="

# 1. Cache Bereinigung (Vorsichtsmaßnahme gegen defekte Downloads)
CACHE_DIR="/netscratch/$USER/.cache/huggingface/hub"
OVIS_DIR="$CACHE_DIR/models--AIDC-AI--Ovis2.5-9B"

if [[ -d "$OVIS_DIR" ]]; then
    echo "⚠️  [WRAPPER] Lösche Cache Verzeichnis: $OVIS_DIR"
    rm -rf "$OVIS_DIR"
else
    echo "ℹ️  [WRAPPER] Kein alter Cache gefunden (Clean start)."
fi

# 2. Installation (pip)
echo "=== [WRAPPER] Installiere Dependencies... ==="

# Upgrade pip zuerst
python -m pip install --upgrade pip --quiet

# Dependencies installieren
# --user ist im Container oft nicht nötig, aber schadet nicht, falls permissions fehlen
pip install --quiet --upgrade --no-warn-script-location \
    "transformers>=4.46.0" \
    "accelerate" \
    "torchvision" \
    "huggingface_hub" \
    "pillow" \
    "pandas" \
    "openpyxl" \
    "tqdm" \
    "einops" \
    "timm" \
    "sentencepiece" \
    "protobuf"

echo "=== [WRAPPER] Dependencies installiert. ==="
echo "=== [WRAPPER] Check Versions: ==="
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"

# 3. Python Skript starten
echo "=== [WRAPPER] Starte Benchmark Skript... ==="
# Wir übergeben alle Argumente ($@) weiter an python, falls nötig
# Hier rufen wir direkt das Skript auf, das im gleichen Repo liegt

PYTHON_SCRIPT="src/eval/models/test.py"

if [[ ! -f "$PYTHON_SCRIPT" ]]; then
    echo "❌ FEHLER: Python Skript nicht gefunden unter: $PYTHON_SCRIPT"
    echo "Aktueller Inhalt von $(pwd):"
    ls -R
    exit 1
fi

python "$PYTHON_SCRIPT"

echo "=== [WRAPPER] Job erfolgreich beendet. ==="