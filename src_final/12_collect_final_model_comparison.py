from pathlib import Path
import pandas as pd

ROOT = Path.cwd()
OUT = ROOT / "results_final6" / "final_model_comparison"
OUT.mkdir(parents=True, exist_ok=True)

candidate_files = [
    ROOT / "results_final6" / "geneformer_final" / "geneformer_locked_test_metrics.csv",
    ROOT / "results_final6" / "pca_scvi_final8" / "pca_scvi_locked_test_results.csv",
    ROOT / "results_final6" / "scgpt_final" / "scgpt_locked_test_results.csv",
]

tables = []

for f in candidate_files:
    if not f.exists():
        print("Missing, skipping:", f)
        continue

    df = pd.read_csv(f)
    df["source_file"] = str(f)

    # Normalize naming if possible.
    if "model" not in df.columns:
        if "geneformer" in str(f).lower():
            df["model"] = "Geneformer"
        elif "scgpt" in str(f).lower():
            df["model"] = "scGPT"
        elif "pca_scvi" in str(f).lower():
            df["model"] = "PCA/scVI"

    tables.append(df)

if not tables:
    raise SystemExit("No result tables found.")

combined = pd.concat(tables, ignore_index=True, sort=False)
combined.to_csv(OUT / "combined_final_model_comparison_raw.csv", index=False)

keep_cols = [c for c in [
    "model",
    "dataset",
    "n",
    "positive_n",
    "negative_n",
    "auroc",
    "auprc",
    "balanced_accuracy",
    "f1",
    "precision",
    "recall",
    "source_file",
] if c in combined.columns]

summary = combined[keep_cols].copy()
summary.to_csv(OUT / "combined_final_model_comparison_summary.csv", index=False)

print("\nCombined summary:")
print(summary.to_string(index=False))
print("\nWrote:")
print(OUT / "combined_final_model_comparison_summary.csv")
