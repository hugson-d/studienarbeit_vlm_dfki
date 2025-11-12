# studienarbeit_vlm_dfki

Tooling to extract, verify, and analyse Kaenguru contest solutions from the
reference PDF into a structured JSON dataset.

## Prerequisites

- Install [uv](https://docs.astral.sh/uv/) (>=0.5) for Python tooling and
  dependency management. On macOS you can run `brew install uv` or follow the
  official install guide.

## Environment Setup

```bash
uv sync
```

`uv sync` creates the `.venv/` virtual environment and installs the Python
dependencies declared in `pyproject.toml`. Activate the environment when you
need an interactive shell:

```bash
source .venv/bin/activate
```

You can also execute scripts without activating the shell by prefixing commands
with `uv run`.

## Common Tasks

- Rebuild the `lösungen.json` dataset:

  ```bash
  uv run python src/create_solutions_from_pdf.py
  ```

- Verify the JSON data against the PDF source:

  ```bash
  uv run python src/verify_solutions.py
  ```

- Analyse the solutions dataset:

  ```bash
  uv run python src/analyze_solutions.py
  ```

All scripts expect `kaenguru_loesungen_alle.pdf` and the generated
`lösungen.json` to live in the repository root.
