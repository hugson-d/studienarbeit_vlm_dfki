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
sbatch scripts/slurm_qwen2_5_vl_3b.sh
```

## Hinweise

- `.env`, `secrets.sh` und `~/.hf_token` sind alle in `.gitignore`
- Der Token wird beim Job-Start automatisch aus diesen Dateien geladen
- Prüfe mit `squeue -u $USER` ob der Job läuft
