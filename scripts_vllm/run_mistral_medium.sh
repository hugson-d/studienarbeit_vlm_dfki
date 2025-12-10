#!/bin/bash
#SBATCH --job-name=vlm_mistral_medium
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=2
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

# ------------------------------
# Pfade und Umgebungsvariablen
# ------------------------------
PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
if [[ "$(basename "$PROJECT_ROOT")" == "scripts_vllm" ]]; then
    PROJECT_ROOT="$(dirname "$PROJECT_ROOT")"
fi

if [[ ! -f "$PROJECT_ROOT/dataset_final.json" ]]; then
    echo "❌ dataset_final.json nicht gefunden. Bitte aus dem Repo-Root starten."
    exit 1
fi

# Caches auf /netscratch (schneller + mehr Platz)
export PIP_CACHE_DIR="/netscratch/$USER/.cache/pip"
mkdir -p "$PIP_CACHE_DIR"

# API Keys laden
for SECRET_FILE in "$PROJECT_ROOT/.env" "$HOME/.hf_token"; do
    if [[ -f "$SECRET_FILE" ]]; then
        set -a
        source "$SECRET_FILE"
        set +a
        break
    fi
done

# MISTRAL_API_KEY prüfen
if [[ -z "${MISTRAL_API_KEY:-}" ]]; then
    echo "⚠️ MISTRAL_API_KEY nicht gesetzt!"
    exit 1
else
    echo "✅ MISTRAL_API_KEY gefunden"
fi

export VLM_PROJECT_ROOT="$PROJECT_ROOT"
export PYTHONUNBUFFERED=1

echo "=========================================="
echo "🚀 VLM Benchmark: Mistral Medium (API)"
echo "PROJECT_ROOT: $PROJECT_ROOT"
echo "=========================================="

# ------------------------------
# Container mit venv + Installation starten
# ------------------------------
srun \
    --container-image=/enroot/nvcr.io_nvidia_pytorch_23.12-py3.sqsh \
    --container-mounts=/netscratch:/netscratch,/ds:/ds:ro,"$PROJECT_ROOT":"$PROJECT_ROOT" \
    --container-workdir="$PROJECT_ROOT" \
    bash -c '
        echo "📦 Erstelle venv und installiere Dependencies..."
        # Venv erstellen (falls nicht vorhanden)
        VENV_PATH="/netscratch/$USER/.venv/mistral_medium"
        if [[ ! -d "$VENV_PATH" ]]; then
            python -m venv "$VENV_PATH"
            echo "✅ Venv erstellt: $VENV_PATH"
        fi
        # Venv aktivieren
        source "$VENV_PATH/bin/activate"
        # Dependencies installieren
        pip install --upgrade pip
        pip install -q "mistralai>=1.9.0" "pydantic>=2.0" "python-dotenv>=1.0" "pandas" "tqdm" "pillow>=10.0"
        echo "✅ Installation abgeschlossen"
        echo "DEBUG: Python: $(which python)"

        # ------------------------------
        # Skript ausführen
        # ------------------------------
        SCRIPT_PATH="'"$PROJECT_ROOT"'/src/eval/vllm_models/run_mistral_medium.py"

        if [[ ! -f "$SCRIPT_PATH" ]]; then
            echo "❌ Python-Skript nicht gefunden: $SCRIPT_PATH"
            exit 1
        fi

        echo "▶️ Starte Mistral Medium Evaluation..."
        python3 "$SCRIPT_PATH"
    '

echo "✅ Job beendet."