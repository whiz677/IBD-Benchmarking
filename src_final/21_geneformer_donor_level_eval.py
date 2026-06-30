from pathlib import Path
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score, f1_score, accuracy_score

PRED_DIR = Path("results_final6/geneformer_final")
OUT = Path("results_final6/geneformer_final/donor_level")
OUT.mkdir(parents=True, exist_ok=True)

files = {
    "train_dev_internal": PRED_DIR / "train_dev_internal_geneformer_predictions.csv",
    "locked_test_martin": PRED_DIR / "locked_test_martin_geneformer_predictions.csv",
    "locked_test_oliver_no_martin": PRED_DIR / "locked_test_oliver_no_martin_geneformer_predictions.csv",
    "locked_test_martin_epithelial": PRED_DIR / "locked_test_martin_epithelial_geneformer_predictions.csv",
    "locked_test_oliver_no_martin_epithelial": PRED_DIR / "locked_test_oliver_no_martin_epithelial_geneformer_predictions.csv",
}

possible_group_cols = [
    "donor_label",
    "sample_label",
    "sample_id",
    "patient_id",
    "source_id",
]

rows = []

for dataset, path in files.items():
    if not path.exists():
        print("Missing:", path)
        continue

    df = pd.read_csv(path)
    group_col = None
    for c in possible_group_cols:
        if c in df.columns and df[c].notna().sum() > 0:
            group_col = c
            break

    if group_col is None:
        print("No donor/sample column found for", dataset, "columns:", list(df.columns))
        continue

    print("\n", dataset, "using group_col =", group_col)

    agg = (
        df.groupby(group_col)
        .agg(
            y_ibd=("y_ibd", lambda x: int(round(x.mean()))),
            mean_prob=("geneformer_prob_ibd", "mean"),
            median_prob=("geneformer_prob_ibd", "median"),
            n_cells=("geneformer_prob_ibd", "size"),
        )
        .reset_index()
    )

    agg["pred"] = (agg["mean_prob"] >= 0.5).astype(int)
    agg.to_csv(OUT / f"{dataset}_donor_level_predictions.csv", index=False)

    if agg["y_ibd"].nunique() < 2:
        print("Skipping metrics; only one class at donor level.")
        continue

    rows.append({
        "dataset": dataset,
        "group_col": group_col,
        "n_groups": len(agg),
        "positive_groups": int((agg["y_ibd"] == 1).sum()),
        "negative_groups": int((agg["y_ibd"] == 0).sum()),
        "auroc": roc_auc_score(agg["y_ibd"], agg["mean_prob"]),
        "auprc": average_precision_score(agg["y_ibd"], agg["mean_prob"]),
        "accuracy": accuracy_score(agg["y_ibd"], agg["pred"]),
        "balanced_accuracy": balanced_accuracy_score(agg["y_ibd"], agg["pred"]),
        "f1": f1_score(agg["y_ibd"], agg["pred"], zero_division=0),
    })

out = pd.DataFrame(rows)
out.to_csv(OUT / "geneformer_donor_level_metrics.csv", index=False)

print("\nDonor/sample-level Geneformer metrics:")
print(out.to_string(index=False))
print("\nWrote:", OUT / "geneformer_donor_level_metrics.csv")
