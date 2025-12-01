#!/bin/bash
#SBATCH --job-name=setup_venv
#SBATCH --partition=H100
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=setup_venv_%j.out
#SBATCH --error=setup_venv_%j.err

# ============================================
# Einmalige Einrichtung der Python-Umgebung
# Ausführen mit: sbatch scripts/setup_venv.sh
# ============================================

PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
if [[ "$(basename $PROJECT_ROOT)" == "scripts" ]]; then
    PROJECT_ROOT="$(dirname $PROJECT_ROOT)"
fi

# WICHTIG: venv auf /netscratch speichern (nicht $HOME oder Projekt-Root!)
VENV_PATH="/netscratch/$USER/vlm_venv"

echo "=========================================="
echo "🔧 Setup Python Virtual Environment"
echo "PROJECT_ROOT: $PROJECT_ROOT"
echo "VENV_PATH: $VENV_PATH"
echo "=========================================="

# HF_TOKEN laden
for SECRET_FILE in "$PROJECT_ROOT/.env" "$PROJECT_ROOT/secrets.sh" "$HOME/.hf_token"; do
    if [ -f "$SECRET_FILE" ]; then
        set -a
        source "$SECRET_FILE"
        set +a
        break
    fi
done

export HF_TOKEN

# Container starten
srun \
  --container-image=/enroot/nvcr.io_nvidia_pytorch_23.12-py3.sqsh \
  --container-mounts=/netscratch:/netscratch,/ds:/ds:ro,"$PROJECT_ROOT":"$PROJECT_ROOT" \
  --export=ALL,HF_TOKEN \
  bash -c "
    echo '=========================================='
    echo '🐍 Erstelle Virtual Environment...'
    echo '=========================================='
    
    cd $PROJECT_ROOT
    
    # Falls .venv existiert, löschen
    if [ -d '$VENV_PATH' ]; then
        echo '⚠️ Lösche bestehende .venv...'
        rm -rf '$VENV_PATH'
    fi
    
    # Neue venv erstellen
    python -m venv '$VENV_PATH'
    source '$VENV_PATH/bin/activate'
    
    echo 'Python: '\$(which python)
    echo 'Version: '\$(python --version)
    
    # Basis-Pakete upgraden
    pip install --upgrade pip wheel setuptools
    
    echo ''
    echo '📦 Installiere Pakete...'
    
    # NumPy 1.x (Container-Kompatibilität)
    pip install 'numpy<2'
    
    # PyTorch (bereits im Container, aber für venv)
    pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu121
    
    # Transformers von GitHub (für Qwen2.5-VL Support)
    pip install 'git+https://github.com/huggingface/transformers'
    
    # VLM Dependencies
    pip install \
      'accelerate>=0.34.0' \
      'qwen-vl-utils[decord]>=0.0.8' \
      'bitsandbytes>=0.43.0' \
      'pillow>=10.0.0' \
      'pydantic>=2.0.0' \
      'pandas<1.6' \
      'openpyxl>=3.1.0' \
      'python-dotenv>=1.0.0' \
      'huggingface_hub>=0.24.0' \
      'tqdm>=4.66.0' \
      'safetensors>=0.4.0' \
      'tokenizers>=0.19.0'
    
    # Flash Attention (optional)
    pip install flash-attn --no-build-isolation 2>/dev/null || echo '⚠️ Flash Attention nicht installiert'
    
    echo ''
    echo '=========================================='
    echo '✅ Setup abgeschlossen!'
    echo '=========================================='
    pip list | head -30
    echo '...'
    echo ''
    echo 'Pakete insgesamt: '\$(pip list | wc -l)
  "
