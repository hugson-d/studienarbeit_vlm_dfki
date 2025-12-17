#!/usr/bin/env python3
"""
Scans JSONL files in `results_vllm` whose filenames contain `n1` or `n5`.
If a JSON object is missing the `model` field (or it's null/empty), adds
`model` with the value derived from the filename (filename without trailing
`_results.jsonl`). Modifies files in-place by default.

Usage:
  python3 add_model_field_to_jsonl.py [--dry-run] [--dir results_vllm]

"""
import json
from pathlib import Path
import argparse


def build_file_list(root: Path):
    files = sorted(set(root.glob("*n1*.jsonl")) | set(root.glob("*n5*.jsonl")))
    return [p for p in files if p.is_file()]


def model_name_from_filename(path: Path) -> str:
    name = path.name
    if name.endswith("_results.jsonl"):
        return name[:-len("_results.jsonl")]
    # fallback: remove extension
    return path.stem


def process_file(path: Path, dry_run: bool = False):
    model_name = model_name_from_filename(path)
    text = path.read_text(encoding='utf-8')
    lines = text.splitlines()

    out_lines = []
    modified = 0

    for idx, line in enumerate(lines, start=1):
        if not line.strip():
            out_lines.append("")
            continue
        try:
            obj = json.loads(line)
        except Exception as e:
            print(f"Warning: could not parse {path.name} line {idx}: {e}")
            out_lines.append(line)
            continue

        if ('model' not in obj) or (obj.get('model') in (None, "")):
            obj['model'] = model_name
            modified += 1

        out_lines.append(json.dumps(obj, ensure_ascii=False))

    if modified:
        if dry_run:
            print(f"{path.name}: would add/overwrite 'model' in {modified} entries (dry-run)")
        else:
            path.write_text('\n'.join(out_lines) + '\n', encoding='utf-8')
            print(f"{path.name}: updated {modified} entries")
    else:
        print(f"{path.name}: no changes needed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', '-d', default='results_vllm', help='Directory containing jsonl files')
    parser.add_argument('--dry-run', action='store_true', help='Do not write changes; only report')
    args = parser.parse_args()

    root = Path(args.dir)
    if not root.exists():
        print(f"Error: directory {root} does not exist")
        return

    files = build_file_list(root)
    if not files:
        print("No files found matching *n1*.jsonl or *n5*.jsonl in the directory")
        return

    total_files = 0
    total_changes = 0

    for p in files:
        total_files += 1
        # Capture printed output per file; process_file prints summary
        process_file(p, dry_run=args.dry_run)

    print(f"Done. Processed {total_files} files.")


if __name__ == '__main__':
    main()
