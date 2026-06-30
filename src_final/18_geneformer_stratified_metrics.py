from pathlib import Path
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score, f1_score

PRED_DIR = Path("results_final6/geneformer_final")
OUT = Path("results_final6/geneformer_final/diagnostics")
OUT.mkdir(parents=True, exist_ok=True)

files = {
    "martin": PRED_DIR / "locked_test_martin_geneformer_predictions.csv",
    "oliver_no_martin": PRED_DIR / "locked_test_oliver_no_martin_geneformer_predictions.csv",
}

rows = []

for dataset, path in files.items():
    df = pd.read_csv(path)

    for group_col in ["broad_cell_group", "cell_type_label", "tissue_label"]:
        if group_col not in df.columns:
            continue

        for group, sub in df.groupby(group_col):
            if len(sub) < 50:
                continue
            if sub["y_ibd"].nunique() < 2:
                continue

            y = sub["y_ibd"].astype(int).values
            prob = sub["geneformer_prob_ibd"].values
            pred = (prob >= 0.5).astype(int)

            rows.append({
                "dataset": dataset,
                "group_col": group_col,
                "group": group,
                "n": len(sub),
                "positive_n": int((sub["y_ibd"] == 1).sum()),
                "negative_n": int((sub["y_ibd"] == 0).sum()),
                "auroc": roc_auc_score(y, prob),
                "auprc": average_precision_score(y, prob),
                "balanced_accuracy": balanced_accuracy_score(y, pred),
                "f1": f1_score(y, pred, zero_division=0),
            })

out = pd.DataFrame(rows)
out = out.sort_values(["dataset", "group_col", "auroc"])
out.to_csv(OUT / "geneformer_stratified_metrics.csv", index=False)

print(out.to_string(index=False))
print("\nWrote:", OUT / "geneformer_stratified_metrics.csv")
