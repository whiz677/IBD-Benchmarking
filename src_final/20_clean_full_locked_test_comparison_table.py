from pathlib import Path
import pandas as pd

ROOT = Path.cwd()
OUT = ROOT / "results_final6" / "final_model_comparison"
OUT.mkdir(parents=True, exist_ok=True)

rows = []

def add_missing_cols(df, cols):
    for c in cols:
        if c not in df.columns:
            df[c] = pd.NA
    return df

# Geneformer all-cells
gf_all_path = ROOT / "results_final6/geneformer_final/geneformer_locked_test_metrics.csv"
if gf_all_path.exists():
    gf_all = pd.read_csv(gf_all_path)
    gf_all = gf_all[gf_all["dataset"].isin(["locked_test_martin", "locked_test_oliver_no_martin"])].copy()
    gf_all["group"] = "all_cells"
    gf_all["test_dataset"] = gf_all["dataset"].replace({
        "locked_test_martin": "martin",
        "locked_test_oliver_no_martin": "oliver_no_martin",
    })
    gf_all["n_test"] = gf_all["n"]
    gf_all["model"] = "Geneformer_frozen"
    rows.append(gf_all)

# Geneformer epithelial
gf_epi_path = ROOT / "results_final6/geneformer_final/geneformer_epithelial_locked_test_metrics.csv"
if gf_epi_path.exists():
    gf_epi = pd.read_csv(gf_epi_path)
    gf_epi = gf_epi[gf_epi["dataset"].isin([
        "locked_test_martin_epithelial",
        "locked_test_oliver_no_martin_epithelial",
    ])].copy()
    gf_epi["group"] = "epithelial"
    gf_epi["test_dataset"] = gf_epi["dataset"].replace({
        "locked_test_martin_epithelial": "martin",
        "locked_test_oliver_no_martin_epithelial": "oliver_no_martin",
    })
    gf_epi["n_test"] = gf_epi["n"]
    gf_epi["model"] = "Geneformer_frozen"
    rows.append(gf_epi)

# PCA/scVI
pca_path = ROOT / "results_final6/pca_scvi_final8/pca_scvi_locked_test_results.csv"
if pca_path.exists():
    rows.append(pd.read_csv(pca_path))
else:
    print("Missing:", pca_path)

# scGPT
scgpt_path = ROOT / "results_final6/scgpt_final/scgpt_locked_test_results.csv"
if scgpt_path.exists():
    rows.append(pd.read_csv(scgpt_path))
else:
    print("Missing:", scgpt_path)

if not rows:
    raise SystemExit("No result tables found.")

combined = pd.concat(rows, ignore_index=True, sort=False)

needed = [
    "model", "group", "test_dataset", "n_test",
    "auroc", "auprc", "balanced_accuracy", "accuracy", "f1",
    "precision", "recall",
]
combined = add_missing_cols(combined, needed)
combined = combined[needed].copy()

combined = combined[combined["test_dataset"].isin(["martin", "oliver_no_martin"])]
combined = combined.sort_values(["group", "test_dataset", "auroc"], ascending=[True, True, False])

out = OUT / "locked_test_full_clean_comparison.csv"
combined.to_csv(out, index=False)

print(combined.to_string(index=False))
print("\nWrote:", out)
