#!/bin/bash
# ============================================================================
# Setup Script für Linux Cluster
# Führt alle notwendigen Schritte aus, um das Benchmark-Skript auszuführen
# ============================================================================

set -e  # Bei Fehler abbrechen

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "=============================================="
echo "🚀 VLM Benchmark Setup für Linux Cluster"
echo "=============================================="
echo "Projekt-Root: $PROJECT_ROOT"
echo ""

# --- 1. UV installieren (falls nicht vorhanden) ---
if ! command -v uv &> /dev/null; then
    echo "📦 Installiere uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
else
    echo "✅ uv bereits installiert: $(uv --version)"
fi

# --- 2. In Projektverzeichnis wechseln ---
cd "$PROJECT_ROOT"

# --- 3. Virtual Environment erstellen/aktivieren ---
if [ ! -d ".venv" ]; then
    echo "🐍 Erstelle Virtual Environment..."
    uv venv .venv
fi

echo "🔄 Aktiviere Virtual Environment..."
source .venv/bin/activate

# --- 4. Dependencies installieren ---
echo "📦 Installiere Python-Pakete..."
uv pip install --upgrade \
    torch \
    transformers \
    accelerate \
    bitsandbytes \
    pillow \
    pydantic \
    pandas \
    openpyxl \
    python-dotenv \
    huggingface_hub \
    tqdm

# --- 5. Flash Attention installieren (optional, für Performance) ---
echo "⚡ Versuche Flash Attention zu installieren..."
uv pip install flash-attn --no-build-isolation 2>/dev/null || echo "⚠️ Flash Attention konnte nicht installiert werden (optional)"

# --- 6. .env Datei prüfen ---
if [ ! -f ".env" ]; then
    echo ""
    echo "⚠️  WARNUNG: .env Datei nicht gefunden!"
    echo "   Erstelle eine .env Datei mit deinem HuggingFace Token:"
    echo "   echo 'HF_TOKEN=hf_xxx...' > .env"
    echo ""
fi

# --- 7. GPU Check ---
echo ""
echo "🖥️ GPU Status:"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
else
    echo "⚠️ nvidia-smi nicht gefunden. GPU-Unterstützung prüfen!"
fi

echo ""
echo "=============================================="
echo "✅ Setup abgeschlossen!"
echo "=============================================="
echo ""
echo "Nächste Schritte:"
echo "  1. Falls noch nicht geschehen: HF_TOKEN in .env setzen"
echo "  2. Alle Jobs starten mit:"
echo "     ./scripts/submit_all_jobs.sh"
echo ""
echo "  Oder einzelnes Modell testen:"
echo "     source .venv/bin/activate"
echo "     python src/eval/models/run_qwen2_5_vl_3b.py"
echo ""
