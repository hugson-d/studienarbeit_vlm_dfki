# Cluster Setup für VLM Evaluation

## HuggingFace Token einrichten

Da der Token nicht im Git-Repository gespeichert werden kann, musst du ihn auf dem Cluster manuell einrichten.

### Option 1: .env Datei im Projekt (empfohlen)

```bash
# Auf dem Cluster, im Projekt-Root:
cd /pfad/zu/studienarbeit_vlm_dfki
echo 'HF_TOKEN="dein_token_hier"' > .env
```

### Option 2: Globale Token-Datei im Home-Verzeichnis

```bash
# Einmalig einrichten:
echo 'HF_TOKEN="dein_token_hier"' > ~/.hf_token
```

### Option 3: secrets.sh Datei

```bash
# Im Projekt-Root:
echo 'export HF_TOKEN="dein_token_hier"' > secrets.sh
```

## Token verifizieren

```bash
source .env && echo "Token: ${HF_TOKEN:0:10}..."
```

## Jobs starten

```bash
# Immer aus dem Projekt-Root starten:
cd /pfad/zu/studienarbeit_vlm_dfki

# Qwen2.5-VL-3B Benchmark auf H100/H200 starten (erstellt venv automatisch auf /netscratch)
# Für H200 ggf. Partition per Flag überschreiben: sbatch -p H200 scripts/run_qwen2_5_vl_3b.sh
sbatch scripts/run_qwen2_5_vl_3b.sh
```

### Schritt-für-Schritt: `run_qwen2_5_vl_3b.sh` ausführen

1. **Auf den Login-Knoten wechseln und ins Projektverzeichnis gehen**
   ```bash
   cd /pfad/zu/studienarbeit_vlm_dfki
   ```
2. **Hugging Face Token bereitstellen** (eine der Optionen oben wählen, z. B. `.env` im Projekt):
   ```bash
   echo 'HF_TOKEN="dein_token_hier"' > .env
   ```
3. **Job absenden** – standardmäßig auf H100, für H200 Partition überschreiben:
   ```bash
   sbatch scripts/run_qwen2_5_vl_3b.sh          # H100
   sbatch -p H200 scripts/run_qwen2_5_vl_3b.sh  # H200
   ```
4. **Status prüfen**
   ```bash
   squeue -u $USER
   ```
5. **Logs ansehen** (werden im Submit-Verzeichnis als `vlm_qwen2_5_vl_3b_<jobid>.out/.err` abgelegt):
   ```bash
   tail -f vlm_qwen2_5_vl_3b_*.out
   ```

## Hinweise

- `.env`, `secrets.sh` und `~/.hf_token` sind alle in `.gitignore`
- Der Token wird beim Job-Start automatisch aus diesen Dateien geladen
- Prüfe mit `squeue -u $USER` ob der Job läuft
- Das SLURM-Skript nutzt den NVIDIA PyTorch 23.12 Container, legt ein venv auf `/netscratch/$USER/vlm_qwen2_5_vl_3b` an (Python 3.10) und installiert die benötigten Pakete inkl. CUDA 12.1 Torch-Build.
