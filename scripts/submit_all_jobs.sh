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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "🚀 Submitting all VLM Benchmark Jobs"
echo "====================================="

for script in "$SCRIPT_DIR"/slurm_*.sh; do
    if [ -f "$script" ]; then
        model_name=$(basename "$script" .sh | sed 's/slurm_//')
        JOB_ID=$(sbatch --parsable "$script")
        echo "  ✅ $model_name -> Job $JOB_ID"
    fi
done

echo ""
echo "====================================="
echo "📊 Status prüfen:    squeue -u \$USER"
echo "📁 Logs:             evaluation_results/logs/"
echo "📈 Nach Abschluss:   python src/eval/combine_results.py"
echo "====================================="
