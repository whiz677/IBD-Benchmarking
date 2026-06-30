from pathlib import Path
import pandas as pd

ROOT = Path.cwd()
OUT = ROOT / "results_final6" / "final_model_comparison"
OUT.mkdir(parents=True, exist_ok=True)

rows = []

# Geneformer
gf = pd.read_csv(ROOT / "results_final6/geneformer_final/geneformer_locked_test_metrics.csv")
gf = gf[gf["dataset"].isin(["locked_test_martin", "locked_test_oliver_no_martin"])].copy()
gf["group"] = "all_cells"
gf["test_dataset"] = gf["dataset"].replace({
    "locked_test_martin": "martin",
    "locked_test_oliver_no_martin": "oliver_no_martin",
})
gf["n_test"] = gf["n"]
gf["model"] = "Geneformer_frozen"
rows.append(gf[[
    "model", "group", "test_dataset", "n_test",
    "auroc", "auprc", "balanced_accuracy", "accuracy", "f1",
    "precision", "recall"
]])

# PCA / scVI
pca = pd.read_csv(ROOT / "results_final6/pca_scvi_final8/pca_scvi_locked_test_results.csv")
pca = pca[pca["group"] == "all_cells"].copy()
for c in ["precision", "recall"]:
    if c not in pca.columns:
        pca[c] = pd.NA
rows.append(pca[[
    "model", "group", "test_dataset", "n_test",
    "auroc", "auprc", "balanced_accuracy", "accuracy", "f1",
    "precision", "recall"
]])

# scGPT
scgpt = pd.read_csv(ROOT / "results_final6/scgpt_final/scgpt_locked_test_results.csv")
scgpt = scgpt[scgpt["group"] == "all_cells"].copy()
for c in ["precision", "recall"]:
    if c not in scgpt.columns:
        scgpt[c] = pd.NA
rows.append(scgpt[[
    "model", "group", "test_dataset", "n_test",
    "auroc", "auprc", "balanced_accuracy", "accuracy", "f1",
    "precision", "recall"
]])

combined = pd.concat(rows, ignore_index=True)

combined = combined.sort_values(["test_dataset", "auroc"], ascending=[True, False])
combined.to_csv(OUT / "locked_test_all_cells_clean_comparison.csv", index=False)

print(combined.to_string(index=False))
print("\nWrote:", OUT / "locked_test_all_cells_clean_comparison.csv")
