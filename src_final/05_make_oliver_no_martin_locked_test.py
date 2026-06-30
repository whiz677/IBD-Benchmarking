from pathlib import Path
import pandas as pd
import anndata as ad
import numpy as np
import scipy.sparse as sp


ROOT = Path.cwd()

IN_FILE = ROOT / "data" / "processed_final6" / "oliver_locked_test_final.h5ad"

OUT_DIR = ROOT / "data" / "processed_final6"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_FILE = OUT_DIR / "oliver_no_martin_locked_test_final.h5ad"
OUT_OBS = OUT_DIR / "oliver_no_martin_locked_test_final_obs.csv"

REPORT_DIR = ROOT / "results_final6" / "processing_reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_FILE = REPORT_DIR / "05_oliver_no_martin_report.txt"


def make_martin_mask(obs):
    mask = pd.Series(False, index=obs.index)

    for col in ["source_id", "sourceID"]:
        if col in obs.columns:
            s = obs[col].astype(str)
            mask = mask | s.str.contains("GSE134809", case=False, na=False)

    for col in ["source_study", "study"]:
        if col in obs.columns:
            s = obs[col].astype(str)
            mask = mask | s.str.contains("Martin2019", case=False, na=False)

    return mask


def add_geneformer_var_fields(adata):
    adata.var_names = adata.var_names.astype(str)
    adata.var_names_make_unique()

    if "ensembl_id" not in adata.var.columns:
        adata.var["ensembl_id"] = adata.var_names.astype(str)
    else:
        adata.var["ensembl_id"] = adata.var["ensembl_id"].astype(str)

    if "gene_symbol" not in adata.var.columns:
        if "gene_symbols" in adata.var.columns:
            adata.var["gene_symbol"] = adata.var["gene_symbols"].astype(str)
        elif "feature_name" in adata.var.columns:
            adata.var["gene_symbol"] = adata.var["feature_name"].astype(str)
        else:
            adata.var["gene_symbol"] = adata.var_names.astype(str)

    return adata


def safe_series(obs, col, default="unknown"):
    if col in obs.columns:
        return obs[col].astype(str).fillna(default)
    return pd.Series([default] * obs.shape[0], index=obs.index, dtype=str)


def clean_obs_before_write(adata):
    adata.obs["dataset_eval"] = "oliver_no_martin"
    adata.obs["dataset_id"] = "oliver_2024_no_gse134809"
    adata.obs["locked_test_dataset"] = "yes"
    adata.obs["martin_gse134809_removed"] = "yes"

    if "source_id" not in adata.obs.columns and "sourceID" in adata.obs.columns:
        adata.obs["source_id"] = adata.obs["sourceID"].astype(str)
    if "source_study" not in adata.obs.columns and "study" in adata.obs.columns:
        adata.obs["source_study"] = adata.obs["study"].astype(str)

    if "cell_type_label" not in adata.obs.columns:
        adata.obs["cell_type_label"] = safe_series(adata.obs, "cell_type")
    if "donor_label" not in adata.obs.columns:
        if "donorID_unified" in adata.obs.columns:
            adata.obs["donor_label"] = safe_series(adata.obs, "donorID_unified")
        else:
            adata.obs["donor_label"] = safe_series(adata.obs, "donor_id")
    if "sample_label" not in adata.obs.columns:
        adata.obs["sample_label"] = safe_series(adata.obs, "sampleID")
    if "tissue_label" not in adata.obs.columns:
        adata.obs["tissue_label"] = safe_series(adata.obs, "tissue")
    if "assay_label" not in adata.obs.columns:
        adata.obs["assay_label"] = safe_series(adata.obs, "assay")

    if "disease_label_raw" not in adata.obs.columns:
        adata.obs["disease_label_raw"] = safe_series(adata.obs, "disease")
    if "control_vs_disease_raw" not in adata.obs.columns:
        adata.obs["control_vs_disease_raw"] = safe_series(adata.obs, "control_vs_disease")
    if "donor_disease_raw" not in adata.obs.columns:
        adata.obs["donor_disease_raw"] = safe_series(adata.obs, "donor_disease")

    if "n_counts" not in adata.obs.columns:
        if sp.issparse(adata.X):
            adata.obs["n_counts"] = np.asarray(adata.X.sum(axis=1)).ravel()
        else:
            adata.obs["n_counts"] = adata.X.sum(axis=1)

    keep_cols = [
        "dataset_eval",
        "dataset_id",
        "subdataset",
        "locked_test_dataset",
        "martin_gse134809_removed",
        "y_ibd",
        "disease_label_raw",
        "control_vs_disease_raw",
        "donor_disease_raw",
        "cell_type_label",
        "broad_cell_group",
        "donor_label",
        "sample_label",
        "tissue_label",
        "assay_label",
        "source_study",
        "source_id",
        "n_counts",
    ]

    keep_cols = [c for c in keep_cols if c in adata.obs.columns]
    adata.obs = adata.obs[keep_cols].copy()

    for col in adata.obs.columns:
        if col == "y_ibd":
            adata.obs[col] = adata.obs[col].astype(int)
        elif col == "n_counts":
            adata.obs[col] = pd.to_numeric(adata.obs[col], errors="coerce").fillna(0)
        else:
            adata.obs[col] = adata.obs[col].astype(str).fillna("unknown")

    adata.obs_names = adata.obs_names.astype(str)
    adata.var_names = adata.var_names.astype(str)

    return adata


def main():
    lines = []

    def log(x=""):
        print(x)
        lines.append(str(x))

    log("=" * 100)
    log("05_make_oliver_no_martin_locked_test.py")
    log(f"Input: {IN_FILE}")

    if not IN_FILE.exists():
        raise FileNotFoundError(f"Missing input file: {IN_FILE}")

    adata = ad.read_h5ad(IN_FILE)

    log(f"Loaded Oliver processed file: {adata.shape}")
    log(f"Obs columns: {list(adata.obs.columns)}")

    martin_mask = make_martin_mask(adata.obs)

    log("\nMartin/GSE134809 rows to remove:")
    log(martin_mask.value_counts().to_string())

    if martin_mask.sum() == 0:
        log("\nWARNING: No Martin/GSE134809 rows found to remove.")
        log("Output will be same as input except renamed dataset_eval/dataset_id.")

    oliver_clean = adata[~martin_mask.values].copy()

    log("\nOliver after removing Martin/GSE134809:")
    log(f"Shape: {oliver_clean.shape}")

    if "y_ibd" not in oliver_clean.obs.columns:
        raise ValueError("No y_ibd column found. Re-run 03_process_oliver.py first.")

    log("\ny_ibd counts:")
    log(oliver_clean.obs["y_ibd"].value_counts(dropna=False).to_string())

    if "subdataset" in oliver_clean.obs.columns:
        log("\nsubdataset counts:")
        log(oliver_clean.obs["subdataset"].astype(str).value_counts().to_string())

    if "disease_label_raw" in oliver_clean.obs.columns:
        log("\ndisease_label_raw counts:")
        log(oliver_clean.obs["disease_label_raw"].astype(str).value_counts().head(30).to_string())

    if "source_id" in oliver_clean.obs.columns:
        log("\nsource_id counts after removal:")
        log(oliver_clean.obs["source_id"].astype(str).value_counts().head(30).to_string())

    if "source_study" in oliver_clean.obs.columns:
        log("\nsource_study counts after removal:")
        log(oliver_clean.obs["source_study"].astype(str).value_counts().head(30).to_string())

    if "broad_cell_group" in oliver_clean.obs.columns:
        log("\nbroad_cell_group counts:")
        log(oliver_clean.obs["broad_cell_group"].astype(str).value_counts().head(30).to_string())

    # Verify Martin is gone.
    remaining_martin_mask = make_martin_mask(oliver_clean.obs)
    log("\nRemaining Martin/GSE134809 rows after removal:")
    log(str(int(remaining_martin_mask.sum())))

    if remaining_martin_mask.sum() != 0:
        raise ValueError("Martin/GSE134809 rows still remain after filtering. Stop.")

    if oliver_clean.obs["y_ibd"].nunique() < 2:
        raise ValueError("Oliver-no-Martin has only one y_ibd class. Stop.")

    oliver_clean = add_geneformer_var_fields(oliver_clean)
    oliver_clean = clean_obs_before_write(oliver_clean)

    log("\nFinal Oliver-no-Martin standardized:")
    log(f"Shape: {oliver_clean.shape}")
    log("y_ibd counts:")
    log(oliver_clean.obs["y_ibd"].value_counts(dropna=False).to_string())
    log("subdataset counts:")
    log(oliver_clean.obs["subdataset"].value_counts(dropna=False).to_string())
    log("source_id counts:")
    log(oliver_clean.obs["source_id"].value_counts(dropna=False).head(30).to_string())
    log("source_study counts:")
    log(oliver_clean.obs["source_study"].value_counts(dropna=False).head(30).to_string())
    log("broad_cell_group counts:")
    log(oliver_clean.obs["broad_cell_group"].value_counts(dropna=False).head(20).to_string())
    log(f"Donors: {oliver_clean.obs['donor_label'].nunique()}")

    log("\nWriting:")
    log(str(OUT_FILE))
    oliver_clean.write_h5ad(OUT_FILE, compression="gzip")
    oliver_clean.obs.to_csv(OUT_OBS)

    REPORT_FILE.write_text("\n".join(lines), encoding="utf-8")

    log("\nSaved:")
    log(str(OUT_FILE))
    log(str(OUT_OBS))
    log(str(REPORT_FILE))


if __name__ == "__main__":
    main()