#!/bin/bash
#SBATCH --job-name=vlm_qwen2_5_vl_72b
#SBATCH --partition=batch
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --output=evaluation_results/logs/qwen2_5_vl_72b_%j.out
#SBATCH --error=evaluation_results/logs/qwen2_5_vl_72b_%j.err

# Projekt-Root ermitteln (eine Ebene über scripts/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Logs-Verzeichnis erstellen
mkdir -p "$PROJECT_ROOT/evaluation_results/logs"

# HF_TOKEN aus .env laden (falls vorhanden)
if [ -f "$PROJECT_ROOT/.env" ]; then
    export $(grep -v '^#' "$PROJECT_ROOT/.env" | xargs)
fi

srun \
  --container-image=/enroot/nvcr.io_nvidia_pytorch_23.12-py3.sqsh \
  --container-workdir="$PROJECT_ROOT" \
  --container-mounts=/netscratch:/netscratch,/ds:/ds:ro,"$PROJECT_ROOT":"$PROJECT_ROOT" \
  --export=ALL \
  bash -c '
    echo "=========================================="
    echo "🚀 VLM Benchmark: Qwen2.5-VL-72B"
    echo "=========================================="
    echo "Start: $(date)"
    echo "Working Directory: $(pwd)"
    echo "HF_TOKEN: ${HF_TOKEN:+gesetzt}"
    echo "GPU:"
    nvidia-smi --query-gpu=name,memory.total --format=csv
    echo ""
    
    # Python-Pakete installieren (ohne torch, da im Container vorhanden)
    echo "📦 Installiere Python-Pakete..."
    pip install --quiet \
      transformers \
      accelerate \
      bitsandbytes \
      pillow \
      pydantic \
      "pandas<1.6" \
      openpyxl \
      python-dotenv \
      huggingface_hub \
      tqdm
    
    # Flash Attention (optional)
    pip install --quiet flash-attn --no-build-isolation 2>/dev/null || echo "⚠️ Flash Attention nicht installiert (optional)"
    
    echo ""
    echo "🏃 Starte Benchmark..."
    python src/eval/models/run_qwen2_5_vl_72b.py
    
    echo ""
    echo "=========================================="
    echo "✅ Fertig: $(date)"
    echo "=========================================="
  '
