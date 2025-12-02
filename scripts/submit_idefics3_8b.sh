#!/bin/bash
# =============================================================================
# SLURM Job-Submission für Idefics3-8B-Llama3
# =============================================================================

#SBATCH --job-name=idefics3-8b
#SBATCH --output=logs/idefics3_8b_%j.out
#SBATCH --error=logs/idefics3_8b_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=A100-40GB,RTXA6000,RTXA6000-SLT,L40S,H100-SLT-NP,H100,H200,A100-80GB,H100-SLT,A100-PCI,H200-AV,H200-DA,H200-PCI,H200-SDS

# =============================================================================
# UMGEBUNG
# =============================================================================

set -euo pipefail

# Logging
echo "=========================================="
echo "🚀 SLURM Job für Idefics3-8B-Llama3"
echo "=========================================="
echo "Job ID:        $SLURM_JOB_ID"
echo "Node:          $SLURMD_NODENAME"
echo "GPU:           $CUDA_VISIBLE_DEVICES"
echo "Start:         $(date)"
echo ""

# Projekt-Root & Python Environment
export VLM_PROJECT_ROOT="$(pwd)"
export PYTHONPATH="$VLM_PROJECT_ROOT:$PYTHONPATH"

# HuggingFace Token (falls benötigt)
# export HF_TOKEN="your_token_here"

# =============================================================================
# DEPENDENCIES INSTALLIEREN
# =============================================================================

echo "📦 Installiere Dependencies..."
bash scripts/install.sh

# =============================================================================
# MODEL EVALUATION
# =============================================================================

echo ""
echo "🔬 Starte Evaluation: Idefics3-8B-Llama3"
echo "=========================================="

python src/eval/models/run_idefics3_8b.py

# =============================================================================
# CLEANUP
# =============================================================================

echo ""
echo "✅ Job abgeschlossen: $(date)"
echo "=========================================="
