# Cluster Notes

These notes document the intended usage pattern for the SLURM launchers in `scripts/inference/zero_shot/`.

## Secrets

The launchers look for credentials in one of these files:

- `.env` in the repository root
- `~/.hf_token`

Typical variables are:

- `HF_TOKEN` for gated Hugging Face models
- `MISTRAL_API_KEY` for Mistral API runs
- `OPENAI_API_KEY` for OpenAI API runs

## Running a Job

Run jobs from the repository root:

```bash
sbatch scripts/inference/zero_shot/run_qwen2_5_vl_3b_vllm.sh
```

The scripts assume the canonical repo layout, especially:

- `data/final/dataset.json`
- `src/eval/zero_shot/`

## Environment Behavior

The SLURM launchers:

- derive the repository root from the script location when needed
- mount the repository into the container
- create or reuse a virtual environment on `/netscratch/$USER`
- install the required runtime dependencies inside that environment

## Logs

The SLURM `--output` and `--error` files are intentionally not tracked by git. If you need to keep job logs for analysis, store them outside the repository or under a separate local-only directory.
