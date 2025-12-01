#!/usr/bin/env python3
"""
Generiert SLURM-Skripte für alle Modelle.
"""

from pathlib import Path

# Modell-Registry
MODELS = {
    # Qwen2.5-VL Familie
    "Qwen2.5-VL-72B":  {"script": "run_qwen2_5_vl_72b.py", "params_b": 72, "large": True},
    "Qwen2.5-VL-32B":  {"script": "run_qwen2_5_vl_32b.py", "params_b": 32, "large": False},
    "Qwen2.5-VL-7B":   {"script": "run_qwen2_5_vl_7b.py",  "params_b": 7,  "large": False},
    "Qwen2.5-VL-3B":   {"script": "run_qwen2_5_vl_3b.py",  "params_b": 3,  "large": False},
    
    # InternVL3 Familie
    "InternVL3-78B":   {"script": "run_internvl3_78b.py",  "params_b": 78, "large": True},
    "InternVL3-38B":   {"script": "run_internvl3_38b.py",  "params_b": 38, "large": False},
    "InternVL3-14B":   {"script": "run_internvl3_14b.py",  "params_b": 14, "large": False},
    "InternVL3-8B":    {"script": "run_internvl3_8b.py",   "params_b": 8,  "large": False},

    # Ovis2.5 Familie
    "Ovis2.5-9B":      {"script": "run_ovis2_5_9b.py",     "params_b": 9,  "large": False},
    "Ovis2.5-2B":      {"script": "run_ovis2_5_2b.py",     "params_b": 2,  "large": False},
    
    # Ovis2 Familie
    "Ovis2-34B":       {"script": "run_ovis2_34b.py",      "params_b": 34, "large": False},
    "Ovis2-16B":       {"script": "run_ovis2_16b.py",      "params_b": 16, "large": False},
    "Ovis2-8B":        {"script": "run_ovis2_8b.py",       "params_b": 8,  "large": False},
    "Ovis2-4B":        {"script": "run_ovis2_4b.py",       "params_b": 4,  "large": False},
}

SLURM_TEMPLATE = '''#!/bin/bash
#SBATCH --job-name=vlm_{job_name}
#SBATCH --partition=H100
#SBATCH --gres=gpu:1
#SBATCH --mem={mem}G
#SBATCH --cpus-per-task=8
#SBATCH --time={time}:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# ============================================
# WICHTIG: Job aus Projekt-Root starten mit:
#   sbatch scripts/slurm_xxx.sh
# ============================================

# Projekt-Root = SLURM_SUBMIT_DIR (wo sbatch aufgerufen wurde)
PROJECT_ROOT="${{SLURM_SUBMIT_DIR}}"

# Falls aus scripts/ gestartet, eine Ebene hoch
if [[ "$(basename $PROJECT_ROOT)" == "scripts" ]]; then
    PROJECT_ROOT="$(dirname $PROJECT_ROOT)"
fi

echo "=========================================="
echo "PROJECT_ROOT: $PROJECT_ROOT"
echo "=========================================="

# Prüfen ob Projekt existiert
if [[ ! -f "$PROJECT_ROOT/dataset_final.json" ]]; then
    echo "❌ FEHLER: dataset_final.json nicht gefunden in $PROJECT_ROOT"
    echo "Bitte starte den Job aus dem Projekt-Root: sbatch scripts/{script_name}"
    exit 1
fi

# Logs-Verzeichnis erstellen
mkdir -p "$PROJECT_ROOT/evaluation_results/logs"

# Outputs verschieben nach Job-Ende
trap "mv ${{SLURM_JOB_NAME}}_${{SLURM_JOB_ID}}.out $PROJECT_ROOT/evaluation_results/logs/ 2>/dev/null; mv ${{SLURM_JOB_NAME}}_${{SLURM_JOB_ID}}.err $PROJECT_ROOT/evaluation_results/logs/ 2>/dev/null" EXIT

# HF_TOKEN aus .env laden
if [ -f "$PROJECT_ROOT/.env" ]; then
    export $(grep -v '^#' "$PROJECT_ROOT/.env" | xargs)
fi

# Container starten
srun \\
  --container-image=/enroot/nvcr.io_nvidia_pytorch_23.12-py3.sqsh \\
  --container-mounts=/netscratch:/netscratch,/ds:/ds:ro,"$PROJECT_ROOT":"$PROJECT_ROOT" \\
  --export=ALL \\
  bash -c "
    echo '=========================================='
    echo '🚀 VLM Benchmark: {model_name}'
    echo '=========================================='
    echo 'Start:' \\$(date)
    echo 'HF_TOKEN: '\\${{HF_TOKEN:+gesetzt}}
    echo 'GPU:'
    nvidia-smi --query-gpu=name,memory.total --format=csv
    echo ''
    
    # In Projektverzeichnis wechseln
    cd $PROJECT_ROOT
    echo 'Working Directory:' \\$(pwd)
    
    # Python-Pakete installieren (kompatibel mit PyTorch 2.2.0 im Container)
    echo '📦 Installiere Python-Pakete...'
    pip install --quiet --no-warn-script-location \\
      'transformers==4.44.2' \\
      'accelerate==0.33.0' \\
      'bitsandbytes==0.43.3' \\
      'pillow>=10.0.0' \\
      'pydantic>=2.0.0' \\
      'pandas<1.6' \\
      'openpyxl>=3.1.0' \\
      'python-dotenv>=1.0.0' \\
      'huggingface_hub==0.24.7' \\
      'tqdm>=4.66.0' \\
      'safetensors>=0.4.0' \\
      'tokenizers>=0.19.0' \\
      2>&1 | grep -v 'dependency resolver' | grep -v 'incompatible' || true
    
    # Flash Attention (optional)
    pip install --quiet flash-attn --no-build-isolation 2>/dev/null || echo '⚠️ Flash Attention nicht installiert (optional)'
    
    echo ''
    echo '🏃 Starte Benchmark...'
    python $PROJECT_ROOT/src/eval/models/{python_script}
    
    echo ''
    echo '=========================================='
    echo '✅ Fertig:' \\$(date)
    echo '=========================================='
  "
'''


def generate_slurm_scripts():
    """Generiert alle SLURM-Skripte."""
    output_dir = Path(__file__).parent
    
    print("📁 Erstelle SLURM-Skripte...")
    
    for model_name, config in MODELS.items():
        # Job-Name (lowercase, underscores)
        job_name = model_name.lower().replace(".", "_").replace("-", "_")
        
        # Ressourcen basierend auf Modellgröße
        if config["large"]:
            gpu_type = "a100:1"
            mem = 128
            time = 48
        else:
            gpu_type = "1"
            mem = 64
            time = 24
        
        # Template ausfüllen
        script_name = f"slurm_{job_name}.sh"
        script_content = SLURM_TEMPLATE.format(
            job_name=job_name,
            model_name=model_name,
            python_script=config["script"],
            script_name=script_name,
            gpu_type=gpu_type,
            mem=mem,
            time=time
        )
        
        # Skript speichern
        script_path = output_dir / f"slurm_{job_name}.sh"
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(script_content)
        
        # Ausführbar machen
        script_path.chmod(0o755)
        
        quant = "4-Bit" if config["params_b"] > 40 else "FP16"
        gpu_info = "A100, 128GB" if config["large"] else "Standard, 64GB"
        print(f"  ✅ slurm_{job_name}.sh ({config['params_b']}B, {quant}, {gpu_info})")
    
    print(f"\n🎉 {len(MODELS)} SLURM-Skripte erstellt!")
    print("\nAusführung:")
    print("  sbatch scripts/slurm_ovis2_4b.sh")
    print("\nAlle Jobs starten:")
    print("  for f in scripts/slurm_*.sh; do sbatch $f; done")


if __name__ == "__main__":
    generate_slurm_scripts()
