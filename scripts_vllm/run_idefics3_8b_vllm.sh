#!/bin/bash
#SBATCH --job-name=vlm_idefics3_8b_vllm_json
#SBATCH --partition=H100,H200,A100-80GB,H100-SLT,A100-PCI
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
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
export HF_HOME="/netscratch/$USER/.cache/huggingface"
mkdir -p "$PIP_CACHE_DIR" "$HF_HOME"

# HF Token laden
for SECRET_FILE in "$PROJECT_ROOT/.env" "$HOME/.hf_token"; do
    if [[ -f "$SECRET_FILE" ]]; then
        set -a
        source "$SECRET_FILE"
        set +a
        break
    fi
done

if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "⚠️ HF_TOKEN nicht gesetzt. Gated Modelle werden fehlschlagen."
else
    echo "✅ HF_TOKEN geladen"
fi

export VLM_PROJECT_ROOT="$PROJECT_ROOT"
export PYTHONUNBUFFERED=1

# ------------------------------
# Virtual Environment
# ------------------------------
VENV_PATH="/netscratch/$USER/.venv/vllm_qwen"

if [[ ! -d "$VENV_PATH" ]]; then
    echo "❌ Venv nicht gefunden unter: $VENV_PATH"
    echo "Bitte erst das Setup-Skript ausführen oder Pfad anpassen."
    exit 1
fi

echo "🚀 Aktiviere Venv: $VENV_PATH"
source "$VENV_PATH/bin/activate"

# ------------------------------
# Skript ausführen
# ------------------------------
SCRIPT_PATH="$PROJECT_ROOT/src/eval/vllm_models/run_idefics3_8b_vllm.py"

if [[ ! -f "$SCRIPT_PATH" ]]; then
    echo "❌ Python-Skript nicht gefunden: $SCRIPT_PATH"
    exit 1
fi

echo "▶️ Starte Idefics3-8B Evaluation mit vLLM..."
python3 "$SCRIPT_PATH"

echo "✅ Job beendet."
