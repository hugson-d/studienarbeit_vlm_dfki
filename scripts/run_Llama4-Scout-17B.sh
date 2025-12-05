#!/bin/bash
#SBATCH --job-name=llama4-scout-17b
#SBATCH --output=/home/%u/logs/%x_%j.out
#SBATCH --error=/home/%u/logs/%x_%j.err
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=24:00:00

# Logging-Verzeichnis erstellen
mkdir -p ~/logs

echo "=============================================="
echo "Job: $SLURM_JOB_NAME (ID: $SLURM_JOB_ID)"
echo "Node: $SLURM_NODELIST"
echo "Start: $(date)"
echo "=============================================="

# Environment
source ~/.bashrc
conda activate vlm_benchmark

# HuggingFace Token
export HF_TOKEN="${HF_TOKEN:-$(cat ~/.huggingface/token 2>/dev/null)}"

# Projekt-Root setzen
export VLM_PROJECT_ROOT="/home/$(whoami)/studienarbeit_vlm_dfki"

# GPU Info
nvidia-smi

# Skript ausführen
cd $VLM_PROJECT_ROOT
python src/eval/models/run_Llama4-Scout-17B.py

echo "=============================================="
echo "Ende: $(date)"
echo "=============================================="
