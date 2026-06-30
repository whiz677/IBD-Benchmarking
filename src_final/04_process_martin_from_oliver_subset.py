from pathlib import Path
import pandas as pd
import anndata as ad
import numpy as np
import scipy.sparse as sp


ROOT = Path.cwd()

IN_FILE = ROOT / "data" / "processed_final6" / "oliver_locked_test_final.h5ad"

OUT_DIR = ROOT / "data" / "processed_final6"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_FILE = OUT_DIR / "martin_locked_test_final.h5ad"
OUT_OBS = OUT_DIR / "martin_locked_test_final_obs.csv"

REPORT_DIR = ROOT / "results_final6" / "processing_reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_FILE = REPORT_DIR / "04_martin_from_oliver_subset_report.txt"


def add_geneformer_var_fields(adata):
    adata.var_names = adata.var_names.astype(str)
    adata.var_names_make_unique()

    # Oliver var_names are Ensembl IDs.
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


def make_martin_mask(obs):
    """
    Detect Martin/GSE134809 rows from processed Oliver.
    In your processed Oliver obs, these are usually:
      source_id == GSE134809
      source_study == Martin2019
    But this also supports original CELLxGENE names:
      sourceID
      study
    """
    mask = pd.Series(False, index=obs.index)

    candidate_source_cols = ["source_id", "sourceID"]
    candidate_study_cols = ["source_study", "study"]

    for col in candidate_source_cols:
        if col in obs.columns:
            s = obs[col].astype(str)
            mask = mask | s.str.contains("GSE134809", case=False, na=False)

    for col in candidate_study_cols:
        if col in obs.columns:
            s = obs[col].astype(str)
            mask = mask | s.str.contains("Martin2019", case=False, na=False)

    return mask


def clean_obs_before_write(adata):
    # Make sure required columns exist.
    adata.obs["dataset_eval"] = "martin"
    adata.obs["dataset_id"] = "martin_2019_from_oliver_gse134809"
    adata.obs["subdataset"] = "martin_from_oliver_gse134809"
    adata.obs["locked_test_dataset"] = "yes"
    adata.obs["derived_from_oliver"] = "yes"

    # Keep source fields for traceability.
    if "source_id" not in adata.obs.columns and "sourceID" in adata.obs.columns:
        adata.obs["source_id"] = adata.obs["sourceID"].astype(str)
    if "source_study" not in adata.obs.columns and "study" in adata.obs.columns:
        adata.obs["source_study"] = adata.obs["study"].astype(str)

    # Standard fields, preserving any already-created standardized columns.
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

    # Ensure raw disease columns exist.
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
        "derived_from_oliver",
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
    log("04_process_martin_from_oliver_subset.py")
    log(f"Input: {IN_FILE}")

    if not IN_FILE.exists():
        raise FileNotFoundError(f"Missing input file: {IN_FILE}")

    adata = ad.read_h5ad(IN_FILE)

    log(f"Loaded Oliver processed file: {adata.shape}")
    log(f"Obs columns: {list(adata.obs.columns)}")

    mask = make_martin_mask(adata.obs)

    log("\nMartin/GSE134809 mask count:")
    log(mask.value_counts().to_string())

    if mask.sum() == 0:
        raise ValueError(
            "No Martin/GSE134809 rows found. Check source_id/source_study columns in Oliver obs."
        )

    martin = adata[mask.values].copy()

    log("\nMartin subset before cleaning:")
    log(f"Shape: {martin.shape}")

    if "source_id" in martin.obs.columns:
        log("\nsource_id counts:")
        log(martin.obs["source_id"].astype(str).value_counts().head(20).to_string())

    if "source_study" in martin.obs.columns:
        log("\nsource_study counts:")
        log(martin.obs["source_study"].astype(str).value_counts().head(20).to_string())

    if "y_ibd" in martin.obs.columns:
        log("\ny_ibd counts:")
        log(martin.obs["y_ibd"].value_counts(dropna=False).to_string())
    else:
        raise ValueError("Martin subset has no y_ibd column. Re-run 03_process_oliver.py first.")

    if "disease_label_raw" in martin.obs.columns:
        log("\ndisease_label_raw counts:")
        log(martin.obs["disease_label_raw"].astype(str).value_counts().head(30).to_string())

    if "control_vs_disease_raw" in martin.obs.columns:
        log("\ncontrol_vs_disease_raw counts:")
        log(martin.obs["control_vs_disease_raw"].astype(str).value_counts().head(30).to_string())

    if "broad_cell_group" in martin.obs.columns:
        log("\nbroad_cell_group counts:")
        log(martin.obs["broad_cell_group"].astype(str).value_counts().head(30).to_string())

    if "donor_label" in martin.obs.columns:
        log("\nDonors:")
        log(str(martin.obs["donor_label"].nunique()))

    # Do not force fail if only one class, because Martin can also be used as
    # lesion/module validation. But print a warning.
    if martin.obs["y_ibd"].nunique() < 2:
        log("\nWARNING: Martin subset has only one y_ibd class.")
        log("It can be used as biological/lesion validation, but not binary AUROC locked testing.")
    else:
        log("\nMartin subset has both classes. Good for binary locked-test evaluation.")

    martin = add_geneformer_var_fields(martin)
    martin = clean_obs_before_write(martin)

    log("\nFinal Martin standardized:")
    log(f"Shape: {martin.shape}")
    log("y_ibd counts:")
    log(martin.obs["y_ibd"].value_counts(dropna=False).to_string())
    log("source_id counts:")
    log(martin.obs["source_id"].value_counts(dropna=False).head(20).to_string())
    log("source_study counts:")
    log(martin.obs["source_study"].value_counts(dropna=False).head(20).to_string())
    log("broad_cell_group counts:")
    log(martin.obs["broad_cell_group"].value_counts(dropna=False).head(20).to_string())
    log(f"Donors: {martin.obs['donor_label'].nunique()}")

    log("\nWriting:")
    log(str(OUT_FILE))
    martin.write_h5ad(OUT_FILE, compression="gzip")
    martin.obs.to_csv(OUT_OBS)

    REPORT_FILE.write_text("\n".join(lines), encoding="utf-8")

    log("\nSaved:")
    log(str(OUT_FILE))
    log(str(OUT_OBS))
    log(str(REPORT_FILE))


if __name__ == "__main__":
    main()