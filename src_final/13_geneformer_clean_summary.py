from pathlib import Path
import pandas as pd

inp = Path("results_final6/geneformer_final/geneformer_locked_test_metrics.csv")
outdir = Path("results_final6/final_model_comparison")
outdir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(inp)

keep = [
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
    "tn",
    "fp",
    "fn",
    "tp",
]
keep = [c for c in keep if c in df.columns]

clean = df[keep].copy()
clean.to_csv(outdir / "geneformer_clean_summary.csv", index=False)

print(clean.to_string(index=False))
print("\nWrote:", outdir / "geneformer_clean_summary.csv")
