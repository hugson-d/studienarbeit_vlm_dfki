#!/bin/bash
#SBATCH --job-name=vlm_internvl3_14b_failure_analysis
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

echo "=========================================="
echo "🔬 VLM Failure Analysis: InternVL3-14B"
echo "PROJECT_ROOT: $PROJECT_ROOT"
echo "=========================================="

# ------------------------------
# Container mit venv + vLLM Installation starten
# ------------------------------
srun \
    --container-image=/enroot/nvcr.io_nvidia_pytorch_23.12-py3.sqsh \
    --container-mounts=/netscratch:/netscratch,/ds:/ds:ro,"$PROJECT_ROOT":"$PROJECT_ROOT" \
    --container-workdir="$PROJECT_ROOT" \
    bash -c '
        echo "📦 Nutze vorhandenen venv..."
        
        # Nutze den existierenden funktionierenden venv!
        VENV_PATH="/netscratch/dhug/vlm_venv"
        
        if [[ ! -d "$VENV_PATH" ]]; then
            echo "❌ venv nicht gefunden: $VENV_PATH"
            echo "Erstelle neuen venv als Fallback..."
            VENV_PATH="/netscratch/$USER/.venv/vllm_internvl"
            python -m venv "$VENV_PATH"
        else
            echo "✅ Nutze existierenden venv: $VENV_PATH"
        fi
        
        # Venv aktivieren
        source "$VENV_PATH/bin/activate"
        
        # Dependencies installieren (falls nötig)
        pip install --upgrade pip
        
        # vLLM mit Vision Support (>= 0.6.0 für guided_decoding)
        pip install -q "vllm>=0.6.0"
        
        # xgrammar für Structured Output Backend (JSON Schema)
        pip install -q xgrammar
        
        # Zusätzliche Dependencies
        pip install -q \
            "numpy<2.0" \
            "transformers>=4.45.0" \
            "accelerate>=0.33.0" \
            "huggingface_hub>=0.24.0" \
            "pydantic>=2.0" \
            "python-dotenv>=1.0" \
            "pandas" \
            "tqdm" \
            "Pillow"
        
        echo "✅ Python-Pakete installiert"
        python --version
        pip show vllm transformers pydantic
        
        cd "$PROJECT_ROOT"
        
        # WICHTIG: sys.path für Import von src.eval.vllm_models
        export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
        
        echo "🚀 Starte Failure Analysis Script (5 runs per task)..."
        python src/eval/vllm_models/run_internvl3_14b_vllm_cot_voting.py
        
        echo "✅ Failure Analysis abgeschlossen!"
    '

echo ""
echo "=========================================="
echo "✅ Job beendet!"
echo "=========================================="
