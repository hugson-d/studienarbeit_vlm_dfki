#!/bin/bash
# ============================================================================
# Startet alle 14 VLM-Benchmarks als separate SLURM-Jobs
#
# Verwendung:
#   ./scripts/submit_all_jobs.sh
#
# Nach Abschluss aller Jobs:
#   python src/eval/combine_results.py
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
MODELS_DIR="$PROJECT_DIR/src/eval/models"

echo "🚀 Submitting VLM Benchmark Jobs"
echo "================================"
echo "Project: $PROJECT_DIR"
echo "Models:  $MODELS_DIR"
echo ""

# Logs-Verzeichnis erstellen
mkdir -p "$PROJECT_DIR/evaluation_results/logs"

# Kleine Modelle (≤40B) - Standard GPU, 64GB RAM
SMALL_MODELS=(
    "run_qwen2_5_vl_3b.py"
    "run_qwen2_5_vl_7b.py"
    "run_qwen2_5_vl_32b.py"
    "run_internvl3_8b.py"
    "run_internvl3_14b.py"
    "run_internvl3_38b.py"
    "run_ovis2_5_2b.py"
    "run_ovis2_5_9b.py"
    "run_ovis2_4b.py"
    "run_ovis2_8b.py"
    "run_ovis2_16b.py"
    "run_ovis2_34b.py"
)

# Große Modelle (>40B) - A100 + 128GB RAM, 4-Bit Quantisierung
LARGE_MODELS=(
    "run_qwen2_5_vl_72b.py"
    "run_internvl3_78b.py"
)

echo "📦 Kleine Modelle (≤40B, FP16/BF16):"
for script in "${SMALL_MODELS[@]}"; do
    model_name="${script%.py}"
    model_name="${model_name#run_}"
    
    JOB_ID=$(sbatch --parsable \
        --job-name="vlm_${model_name}" \
        --output="$PROJECT_DIR/evaluation_results/logs/${model_name}_%j.out" \
        --error="$PROJECT_DIR/evaluation_results/logs/${model_name}_%j.err" \
        --time=24:00:00 \
        --mem=64G \
        --gres=gpu:1 \
        --wrap="cd $PROJECT_DIR && source .venv/bin/activate && python $MODELS_DIR/$script")
    
    echo "  ✅ $model_name -> Job $JOB_ID"
done

echo ""
echo "📦 Große Modelle (>40B, 4-Bit Quantisierung):"
for script in "${LARGE_MODELS[@]}"; do
    model_name="${script%.py}"
    model_name="${model_name#run_}"
    
    JOB_ID=$(sbatch --parsable \
        --job-name="vlm_${model_name}" \
        --output="$PROJECT_DIR/evaluation_results/logs/${model_name}_%j.out" \
        --error="$PROJECT_DIR/evaluation_results/logs/${model_name}_%j.err" \
        --time=48:00:00 \
        --mem=128G \
        --gres=gpu:a100:1 \
        --wrap="cd $PROJECT_DIR && source .venv/bin/activate && python $MODELS_DIR/$script")
    
    echo "  ✅ $model_name -> Job $JOB_ID"
done

echo ""
echo "================================"
echo "📊 Status prüfen:    squeue -u \$USER"
echo "📁 Logs:             $PROJECT_DIR/evaluation_results/logs/"
echo "📈 Nach Abschluss:   python src/eval/combine_results.py"
echo "================================"
