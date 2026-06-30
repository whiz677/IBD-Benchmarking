from pathlib import Path
import pandas as pd

files = [
    "results_final6/geneformer_final/geneformer_locked_test_metrics.csv",
    "results_final6/pca_scvi_final8/pca_scvi_locked_test_results.csv",
    "results_final6/scgpt_final/scgpt_locked_test_results.csv",
]

for f in files:
    p = Path(f)
    print("\n" + "="*100)
    print("FILE:", p)
    if not p.exists():
        print("MISSING")
        continue

    df = pd.read_csv(p)
    print("shape:", df.shape)
    print("columns:")
    for c in df.columns:
        print(" ", c)

    print("\nhead:")
    print(df.head(20).to_string(index=False))
