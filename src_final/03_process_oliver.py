from pathlib import Path
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp


ROOT = Path.cwd()

RAW_DIR = ROOT / "data" / "raw" / "ibd_oliver_2024_cellxgene" / "manual_download"
OUT_DIR = ROOT / "data" / "processed_final6"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_FILE = OUT_DIR / "oliver_locked_test_final.h5ad"
OUT_OBS = OUT_DIR / "oliver_locked_test_final_obs.csv"

RANDOM_SEED = 0

# Keep enough data but do not let huge normal class dominate.
# Each Oliver subdataset is balanced separately.
MAX_PER_CLASS_PER_SUBDATASET = 12000

rng = np.random.default_rng(RANDOM_SEED)


OLIVER_FILES = {
    "oliver_large_intestine": RAW_DIR / "large_intestine",
    "oliver_small_intestine": RAW_DIR / "small_intestine",
}


def find_one_h5ad(folder: Path) -> Path:
    files = sorted(folder.glob("*.h5ad"))
    if len(files) == 0:
        raise FileNotFoundError(f"No .h5ad found in {folder}")
    if len(files) > 1:
        print(f"WARNING: multiple h5ad files in {folder}; using first:")
        for f in files:
            print("  ", f)
    return files[0]


def disease_to_y(disease_value, control_vs_disease_value=None, donor_disease_value=None):
    """
    Returns:
      0 = normal/control
      1 = IBD/UC/CD/PIBD
      np.nan = exclude other disease, cancer, unknown

    For Oliver:
      Large intestine disease values include:
        normal, colorectal cancer, ulcerative colitis, inflammatory bowel disease
      Small intestine disease values include:
        normal, Crohn disease
    """
    vals = [
        str(disease_value).strip().lower() if disease_value is not None else "",
        str(control_vs_disease_value).strip().lower() if control_vs_disease_value is not None else "",
        str(donor_disease_value).strip().lower() if donor_disease_value is not None else "",
    ]

    text = " | ".join(vals)

    # Exclude cancers completely.
    if "cancer" in text or "colorectal" in text or "tumour" in text or "tumor" in text:
        return np.nan

    # Exclude polyps / other non-IBD diseases if present.
    if "polyp" in text:
        return np.nan

    # Controls.
    if (
        "normal" in text
        or "control" in text
        or "organ_donor" in text
        or "non_pathological" in text
    ):
        # If it also explicitly says IBD, disease should win.
        if any(k in text for k in ["crohn", "ulcerative", "colitis", "ibd", "pibd"]):
            return 1
        return 0

    # IBD disease labels.
    if any(k in text for k in ["crohn", "ulcerative", "colitis", "ibd", "pibd"]):
        return 1

    return np.nan


def make_broad_cell_group(cell_type):
    s = str(cell_type).lower()

    epithelial_terms = [
        "colonocyte",
        "enterocyte",
        "goblet",
        "tuft",
        "paneth",
        "enteroendocrine",
        "crypt",
        "stem cell",
        "transit amplifying",
        "best4",
        "m cell",
        "foveolar",
        "mucous neck",
    ]

    if any(t in s for t in epithelial_terms):
        return "epithelial"

    if any(t in s for t in ["t cell", "nk cell", "b cell", "plasma"]):
        return "lymphoid_plasma"

    if any(t in s for t in ["macrophage", "monocyte", "dendritic", "neutrophil", "myeloid"]):
        return "myeloid"

    if any(t in s for t in ["fibroblast", "stromal", "endothelial", "pericyte", "smooth muscle"]):
        return "stromal_endothelial"

    return "other"


def safe_str_series(obs, col, default="unknown"):
    if col in obs.columns:
        return obs[col].astype(str).fillna(default)
    return pd.Series([default] * obs.shape[0], index=obs.index, dtype=str)


def add_geneformer_var_fields(adata):
    """
    Geneformer needs Ensembl IDs. Oliver var_names are already Ensembl IDs.
    Keep both Ensembl ID and gene symbol columns.
    """
    adata.var_names = adata.var_names.astype(str)
    adata.var_names_make_unique()

    adata.var["ensembl_id"] = adata.var_names.astype(str)

    if "gene_symbols" in adata.var.columns:
        adata.var["gene_symbol"] = adata.var["gene_symbols"].astype(str)
    elif "feature_name" in adata.var.columns:
        adata.var["gene_symbol"] = adata.var["feature_name"].astype(str)
    else:
        adata.var["gene_symbol"] = adata.var_names.astype(str)

    return adata


def clean_obs_before_write(adata):
    keep_cols = [
        "dataset_eval",
        "dataset_id",
        "subdataset",
        "locked_test_dataset",
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


def process_one(subdataset_name, folder):
    h5ad_path = find_one_h5ad(folder)

    print("\n" + "=" * 120)
    print(f"Processing {subdataset_name}")
    print(f"File: {h5ad_path}")

    adata = ad.read_h5ad(h5ad_path)

    print("Original shape:", adata.shape)
    print("Original disease counts:")
    print(adata.obs["disease"].astype(str).value_counts(dropna=False).head(30))

    # Build binary IBD label.
    y = []
    for i in range(adata.n_obs):
        disease = adata.obs["disease"].iloc[i] if "disease" in adata.obs.columns else None
        cvd = adata.obs["control_vs_disease"].iloc[i] if "control_vs_disease" in adata.obs.columns else None
        donor_dis = adata.obs["donor_disease"].iloc[i] if "donor_disease" in adata.obs.columns else None
        y.append(disease_to_y(disease, cvd, donor_dis))

    y = pd.Series(y, index=adata.obs_names)

    print("\nBinary label counts before filtering:")
    print(y.value_counts(dropna=False))

    keep = y.notna().values
    adata = adata[keep].copy()
    y = y.loc[adata.obs_names].astype(int)

    adata.obs["y_ibd"] = y.values

    print("\nAfter excluding cancer/unknown/other:")
    print("Shape:", adata.shape)
    print(adata.obs["y_ibd"].value_counts())

    # Balanced downsample per subdataset.
    chosen = []
    for label in [0, 1]:
        idx = np.where(adata.obs["y_ibd"].values == label)[0]
        if len(idx) == 0:
            print(f"WARNING: no label {label} in {subdataset_name}")
            continue

        n_take = min(len(idx), MAX_PER_CLASS_PER_SUBDATASET)
        chosen.extend(rng.choice(idx, size=n_take, replace=False).tolist())

    chosen = np.array(sorted(chosen), dtype=int)
    adata = adata[chosen].copy()

    print("\nAfter balanced downsampling:")
    print("Shape:", adata.shape)
    print(adata.obs["y_ibd"].value_counts())

    # Standardized obs fields.
    adata.obs["dataset_eval"] = "oliver"
    adata.obs["dataset_id"] = "oliver_2024"
    adata.obs["subdataset"] = subdataset_name
    adata.obs["locked_test_dataset"] = "yes"

    adata.obs["disease_label_raw"] = safe_str_series(adata.obs, "disease")
    adata.obs["control_vs_disease_raw"] = safe_str_series(adata.obs, "control_vs_disease")
    adata.obs["donor_disease_raw"] = safe_str_series(adata.obs, "donor_disease")

    adata.obs["cell_type_label"] = safe_str_series(adata.obs, "cell_type")
    adata.obs["broad_cell_group"] = adata.obs["cell_type_label"].map(make_broad_cell_group).astype(str)

    adata.obs["donor_label"] = safe_str_series(adata.obs, "donorID_unified")
    adata.obs["sample_label"] = safe_str_series(adata.obs, "sampleID")
    adata.obs["tissue_label"] = safe_str_series(adata.obs, "tissue")
    adata.obs["assay_label"] = safe_str_series(adata.obs, "assay")
    adata.obs["source_study"] = safe_str_series(adata.obs, "study")
    adata.obs["source_id"] = safe_str_series(adata.obs, "sourceID")

    # n_counts exists in Oliver, but make sure it is numeric.
    if "n_counts" not in adata.obs.columns:
        if sp.issparse(adata.X):
            adata.obs["n_counts"] = np.asarray(adata.X.sum(axis=1)).ravel()
        else:
            adata.obs["n_counts"] = adata.X.sum(axis=1)

    adata = add_geneformer_var_fields(adata)
    adata = clean_obs_before_write(adata)

    print("\nFinal standardized:")
    print("Shape:", adata.shape)
    print("Disease counts:")
    print(adata.obs["y_ibd"].value_counts())
    print("Broad cell groups:")
    print(adata.obs["broad_cell_group"].value_counts().head(20))
    print("Disease raw:")
    print(adata.obs["disease_label_raw"].value_counts().head(20))
    print("Donors:", adata.obs["donor_label"].nunique())

    return adata


def main():
    pieces = []

    for subdataset_name, folder in OLIVER_FILES.items():
        pieces.append(process_one(subdataset_name, folder))

    print("\n" + "=" * 120)
    print("Concatenating Oliver locked-test subdatasets...")

    combined = ad.concat(
        pieces,
        join="inner",
        index_unique="-",
        label=None,
    )

    combined.var_names_make_unique()
    combined = add_geneformer_var_fields(combined)
    combined = clean_obs_before_write(combined)

    print("\nCombined Oliver final:")
    print("Shape:", combined.shape)
    print("Disease counts:")
    print(combined.obs["y_ibd"].value_counts())
    print("Subdataset counts:")
    print(combined.obs["subdataset"].value_counts())
    print("Raw disease labels:")
    print(combined.obs["disease_label_raw"].value_counts())
    print("Broad groups:")
    print(combined.obs["broad_cell_group"].value_counts())
    print("Donors:", combined.obs["donor_label"].nunique())

    if combined.obs["y_ibd"].nunique() < 2:
        raise ValueError("Oliver output has only one class. Stop.")

    print("\nWriting:")
    print(OUT_FILE)
    combined.write_h5ad(OUT_FILE, compression="gzip")
    combined.obs.to_csv(OUT_OBS)

    print("\nSaved:")
    print(OUT_FILE)
    print(OUT_OBS)


if __name__ == "__main__":
    main()