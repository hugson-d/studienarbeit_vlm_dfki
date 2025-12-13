import json
import csv
import argparse
from collections import defaultdict
from pathlib import Path


def new_model_stats(model_name: str):
    return {
        "model": model_name,
        "total": 0,
        "correct": 0,
        "null_prediction": 0,
        "invalid_formats": 0,          # format_valid == False
        "missing_format_flag": 0,      # format_valid fehlt komplett
        "categories": defaultdict(lambda: {
            "total": 0,
            "correct": 0,
            "null_prediction": 0,
            "invalid_formats": 0,
            "missing_format_flag": 0,
        })
    }


def analyze_results(results_dir: Path, out_csv: Path):
    print("Starte Analyse...")
    if not results_dir.exists():
        print(f"Ordner {results_dir} nicht gefunden!")
        return

    model_data = {}

    for file_path in results_dir.glob("*.jsonl"):
        print(f"Verarbeite {file_path.name}...")
        with open(file_path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    print(f"Fehler beim Parsen in {file_path.name} Zeile {line_no}")
                    continue

                # Modellname: aus Feld oder aus Dateiname ableiten
                model = entry.get("model")
                if not model:
                    filename = file_path.stem
                    model = filename.replace("_results", "").replace("_", "-")
                if not model:
                    continue

                if model not in model_data:
                    model_data[model] = new_model_stats(model)

                cat = entry.get("math_category", "unknown")

                # prediction-Logik: fehlend oder None => null_prediction (zählt als Fehler)
                pred = entry.get("prediction", None)
                is_null_pred = (pred is None)

                # correctness: nur zählen, wenn prediction NICHT null ist
                is_correct = bool(entry.get("is_correct", False))
                count_as_correct = (not is_null_pred) and is_correct

                # format_valid: unabhängig von prediction zählen
                if "format_valid" in entry:
                    format_valid = bool(entry.get("format_valid"))
                    if not format_valid:
                        model_data[model]["invalid_formats"] += 1
                        model_data[model]["categories"][cat]["invalid_formats"] += 1
                else:
                    model_data[model]["missing_format_flag"] += 1
                    model_data[model]["categories"][cat]["missing_format_flag"] += 1

                # Totals
                model_data[model]["total"] += 1
                model_data[model]["categories"][cat]["total"] += 1

                if is_null_pred:
                    model_data[model]["null_prediction"] += 1
                    model_data[model]["categories"][cat]["null_prediction"] += 1

                if count_as_correct:
                    model_data[model]["correct"] += 1
                    model_data[model]["categories"][cat]["correct"] += 1

    if not model_data:
        print("Keine gültigen Einträge gefunden.")
        return

    # CSV schreiben
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", encoding="utf-8", newline="") as csvfile:
        fieldnames = [
            "model", "category",
            "total", "correct", "accuracy",
            "null_prediction", "coverage",
            "invalid_formats", "missing_format_flag"
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for model, data in sorted(model_data.items()):
            total = data["total"]
            correct = data["correct"]
            nulls = data["null_prediction"]
            acc = (correct / total) if total else 0.0
            coverage = ((total - nulls) / total) if total else 0.0

            writer.writerow({
                "model": model,
                "category": "ALL",
                "total": total,
                "correct": correct,
                "accuracy": f"{acc:.4f}",
                "null_prediction": nulls,
                "coverage": f"{coverage:.4f}",
                "invalid_formats": data["invalid_formats"],
                "missing_format_flag": data["missing_format_flag"],
            })

            for cat, cat_data in sorted(
                data["categories"].items(),
                key=lambda x: str(x[0]) if x[0] is not None else ""
            ):
                cat_total = cat_data["total"]
                cat_correct = cat_data["correct"]
                cat_nulls = cat_data["null_prediction"]
                cat_acc = (cat_correct / cat_total) if cat_total else 0.0
                cat_coverage = ((cat_total - cat_nulls) / cat_total) if cat_total else 0.0

                writer.writerow({
                    "model": model,
                    "category": cat,
                    "total": cat_total,
                    "correct": cat_correct,
                    "accuracy": f"{cat_acc:.4f}",
                    "null_prediction": cat_nulls,
                    "coverage": f"{cat_coverage:.4f}",
                    "invalid_formats": cat_data["invalid_formats"],
                    "missing_format_flag": cat_data["missing_format_flag"],
                })

    print(f"Aggregierte CSV gespeichert unter: {out_csv}")

    # Konsolenausgabe
    print("\n" + "=" * 80)
    print("ERGEBNISSE: Accuracy (inkl. null im Nenner) und Format-Validität pro Modell")
    print("=" * 80)

    for model, data in sorted(model_data.items()):
        total = data["total"]
        correct = data["correct"]
        nulls = data["null_prediction"]
        acc = (correct / total) if total else 0.0
        coverage = ((total - nulls) / total) if total else 0.0

        print(f"\nModell: {model}")
        print(f"  Accuracy: {correct}/{total} = {acc:.1%}")
        print(f"  Null-Predictions: {nulls} (Coverage: {coverage:.1%})")
        print(f"  Ungültige Formate (format_valid==False): {data['invalid_formats']}")
        print(f"  Fehlendes format_valid Feld: {data['missing_format_flag']}")

        print("  Pro math_category:")
        for cat, cat_data in sorted(
            data["categories"].items(),
            key=lambda x: str(x[0]) if x[0] is not None else ""
        ):
            ct = cat_data["total"]
            cc = cat_data["correct"]
            cn = cat_data["null_prediction"]
            cacc = (cc / ct) if ct else 0.0
            ccov = ((ct - cn) / ct) if ct else 0.0
            print(
                f"    {cat}: {cc}/{ct} = {cacc:.1%} | null={cn} (cov={ccov:.1%})"
                f" | invalid_fmt={cat_data['invalid_formats']}"
                f" | missing_fmt_flag={cat_data['missing_format_flag']}"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("/Users/dennishug/Desktop/vlm_repo/results_vllm"),
        help="Pfad zum results_vllm Ordner"
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "auswertung_aggregiert.csv",
        help="Pfad für die aggregierte CSV"
    )
    args = parser.parse_args()
    analyze_results(args.results_dir, args.out_csv)