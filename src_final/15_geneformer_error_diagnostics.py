from pathlib import Path
import pandas as pd

PRED_DIR = Path("results_final6/geneformer_final")
OUT = Path("results_final6/geneformer_final/diagnostics")
OUT.mkdir(parents=True, exist_ok=True)

files = [
    "train_dev_internal_geneformer_predictions.csv",
    "locked_test_martin_geneformer_predictions.csv",
    "locked_test_oliver_no_martin_geneformer_predictions.csv",
]

for fname in files:
    p = PRED_DIR / fname
    if not p.exists():
        print("Missing:", p)
        continue

    df = pd.read_csv(p)
    print("\n" + "="*100)
    print(fname)
    print("shape:", df.shape)

    label = "y_ibd"
    prob = "geneformer_prob_ibd"

    print("\nMean predicted IBD probability by true label:")
    print(df.groupby(label)[prob].agg(["count", "mean", "std", "min", "max"]).to_string())

    for group_col in ["broad_cell_group", "cell_type_label", "dataset_id", "dataset_eval", "tissue_label"]:
        if group_col in df.columns:
            print(f"\nMean predicted IBD probability by {group_col} and label:")
            summary = df.groupby([group_col, label])[prob].agg(["count", "mean"]).reset_index()
            print(summary.head(50).to_string(index=False))
            summary.to_csv(OUT / f"{fname.replace('.csv','')}_by_{group_col}.csv", index=False)

print("\nWrote diagnostics to:", OUT)
