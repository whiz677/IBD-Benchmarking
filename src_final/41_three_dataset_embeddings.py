from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import scipy.sparse as sp

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

try:
    import scvi
    HAS_SCVI = True
except Exception as e:
    HAS_SCVI = False
    SCVI_IMPORT_ERROR = e


ROOT = Path.cwd()

OUT_DIR = ROOT / "results" / "three_dataset_benchmark"
OUT_DIR.mkdir(parents=True, exist_ok=True)

KONG_FILE = ROOT / "data" / "processed" / "ibd_kong_2023_processed.h5ad"
SMILLIE_FILE = ROOT / "data" / "processed" / "ibd_smillie_uc_processed.h5ad"

MAX_CELLS_PER_DATASET = 18000
N_HVG = 2500
N_PCA = 30
N_LATENT = 30
SCVI_EPOCHS = 30
RANDOM_SEED = 0

rng = np.random.default_rng(RANDOM_SEED)


def find_garrido_file():
    candidates = [
        ROOT / "data" / "processed" / "ibd_garrido_trigo_processed.h5ad",
        ROOT / "data" / "processed" / "ibd_processed.h5ad",
        ROOT / "data" / "raw" / "cellxgene" / "ibd_garrido_trigo" / "manual_download" / "cellxgene_download.h5ad",
    ]

    for p in candidates:
        if p.exists():
            print(f"Found Garrido file: {p}")
            return p

    raise FileNotFoundError("Could not find Garrido file.")


def make_unique_names(names):
    seen = {}
    out = []
    for name in names:
        name = str(name)
        if name not in seen:
            seen[name] = 0
            out.append(name)
        else:
            seen[name] += 1
            out.append(f"{name}_{seen[name]}")
    return out


def standardize_gene_names(a, name):
    print(f"\nGene-name check for {name}")
    print("First 5 original var_names:", list(a.var_names[:5]))
    print("Var columns:", list(a.var.columns))

    if "feature_name" in a.var.columns:
        genes = a.var["feature_name"].astype(str).values
        print(f"Using feature_name for {name}")
    elif "gene_symbols" in a.var.columns:
        genes = a.var["gene_symbols"].astype(str).values
        print(f"Using gene_symbols for {name}")
    elif "gene_symbol" in a.var.columns:
        genes = a.var["gene_symbol"].astype(str).values
        print(f"Using gene_symbol for {name}")
    else:
        genes = a.var_names.astype(str).values
        print(f"Using existing var_names for {name}")

    a.var_names = make_unique_names(genes)
    a.var_names_make_unique()

    print("First 5 standardized var_names:", list(a.var_names[:5]))
    return a


def disease_to_binary(x):
    s = str(x).strip().lower()

    if s in ["0", "0.0", "false", "no"]:
        return 0
    if s in ["1", "1.0", "true", "yes"]:
        return 1
    if "normal" in s or "healthy" in s or "control" in s:
        return 0
    if "crohn" in s or "ulcerative" in s or "ibd" in s or "colitis" in s or "inflammatory bowel" in s:
        return 1

    return np.nan


def pick_col(obs, candidates, required=True):
    for c in candidates:
        if c in obs.columns:
            return c
    if required:
        raise ValueError(f"Missing required column. Tried {candidates}. Available: {list(obs.columns)}")
    return None


def standardize_obs(a, dataset_name):
    obs = a.obs.copy()

    if "y_ibd" in obs.columns:
        y = obs["y_ibd"].map(disease_to_binary)
    else:
        disease_col = pick_col(obs, ["disease", "diagnosis", "condition", "disease_label_raw"])
        y = obs[disease_col].map(disease_to_binary)

    cell_col = pick_col(obs, ["cell_type_label", "cell_type", "celltype", "author_cell_type"], required=False)
    donor_col = pick_col(obs, ["donor_label", "donor_id", "donor", "sample_id", "subject_id", "patient_id"], required=False)
    tissue_col = pick_col(obs, ["tissue_label", "tissue", "tissue_general", "organ"], required=False)

    a.obs["dataset_eval"] = dataset_name
    a.obs["y_ibd"] = y.values

    a.obs["cell_type_label"] = obs[cell_col].astype(str).values if cell_col else "unknown"
    a.obs["donor_label"] = obs[donor_col].astype(str).values if donor_col else "unknown"
    a.obs["tissue_label"] = obs[tissue_col].astype(str).values if tissue_col else "unknown"

    keep = ~pd.isna(a.obs["y_ibd"])
    a = a[keep].copy()
    a.obs["y_ibd"] = a.obs["y_ibd"].astype(int)

    print(f"\n{dataset_name} standardized:")
    print(a.shape)
    print(a.obs["y_ibd"].value_counts().to_dict())
    print("donors:", a.obs["donor_label"].nunique())

    return a


def balanced_downsample(a, max_cells, dataset_name):
    chosen = []
    y = a.obs["y_ibd"].astype(int).values

    for label in [0, 1]:
        idx = np.where(y == label)[0]
        if len(idx) == 0:
            print(f"WARNING {dataset_name}: no label {label}")
            continue

        n_take = min(len(idx), max_cells // 2)
        chosen.extend(rng.choice(idx, size=n_take, replace=False).tolist())

    chosen = np.array(sorted(chosen))

    out = a[chosen].copy()

    print(f"\n{dataset_name} after downsample:")
    print(out.shape)
    print(out.obs["y_ibd"].value_counts().to_dict())

    return out


def assign_broad_group(cell_type):
    s = str(cell_type).lower()

    if any(t in s for t in ["epithelial", "enterocyte", "goblet", "paneth", "brush", "enteroendocrine", "stem cell", "colonocyte"]):
        return "epithelial"
    if any(t in s for t in ["myeloid", "macrophage", "monocyte", "dendritic", "neutrophil"]):
        return "myeloid"
    if any(t in s for t in ["t cell", "b cell", "plasma", "lymphocyte", "natural killer", "nk", "cd4", "cd8"]):
        return "lymphoid_plasma"
    if any(t in s for t in ["stromal", "fibroblast", "myofibroblast", "pericyte", "endothelial", "glial"]):
        return "stromal_endothelial"

    return "other"


def ensure_float32(a):
    if sp.issparse(a.X):
        a.X = a.X.astype(np.float32)
    else:
        a.X = np.asarray(a.X, dtype=np.float32)
    return a


def main():
    if not KONG_FILE.exists():
        raise FileNotFoundError(f"Missing {KONG_FILE}")
    if not SMILLIE_FILE.exists():
        raise FileNotFoundError(f"Missing {SMILLIE_FILE}")

    garrido_file = find_garrido_file()

    print("\nLoading datasets...")
    g = ad.read_h5ad(garrido_file)
    k = ad.read_h5ad(KONG_FILE)
    s = ad.read_h5ad(SMILLIE_FILE)

    g = standardize_gene_names(g, "garrido")
    k = standardize_gene_names(k, "kong")
    s = standardize_gene_names(s, "smillie")

    g = standardize_obs(g, "garrido")
    k = standardize_obs(k, "kong")
    s = standardize_obs(s, "smillie")

    g = balanced_downsample(g, MAX_CELLS_PER_DATASET, "garrido")
    k = balanced_downsample(k, MAX_CELLS_PER_DATASET, "kong")
    s = balanced_downsample(s, MAX_CELLS_PER_DATASET, "smillie")

    common_genes = np.intersect1d(np.intersect1d(g.var_names.astype(str), k.var_names.astype(str)), s.var_names.astype(str))

    print(f"\nCommon genes across 3 datasets: {len(common_genes):,}")

    if len(common_genes) < 1000:
        raise ValueError("Too few common genes across three datasets.")

    g = ensure_float32(g[:, common_genes].copy())
    k = ensure_float32(k[:, common_genes].copy())
    s = ensure_float32(s[:, common_genes].copy())

    combined = ad.concat(
        [g, k, s],
        join="inner",
        index_unique="-",
    )

    combined.var_names_make_unique()
    combined.obs["broad_cell_group"] = combined.obs["cell_type_label"].map(assign_broad_group)

    combined.layers["counts_like"] = combined.X.copy()

    print("\nCombined before normalization:")
    print(combined.shape)
    print(combined.obs.groupby(["dataset_eval", "y_ibd"]).size())

    print("\nNormalizing/log-transforming...")
    sc.pp.normalize_total(combined, target_sum=1e4)
    sc.pp.log1p(combined)

    print("Selecting HVGs...")
    sc.pp.highly_variable_genes(
        combined,
        n_top_genes=N_HVG,
        batch_key="dataset_eval",
        flavor="seurat",
    )

    combined = combined[:, combined.var["highly_variable"]].copy()

    print("\nCombined after HVG:")
    print(combined.shape)

    combined.write_h5ad(OUT_DIR / "combined_garrido_kong_smillie_hvg.h5ad", compression="gzip")
    combined.obs.to_csv(OUT_DIR / "combined_garrido_kong_smillie_obs.csv")

    print("\nRunning PCA...")
    X = combined.X.toarray() if sp.issparse(combined.X) else np.asarray(combined.X)
    X = np.asarray(X, dtype=np.float32)

    pca_pipe = make_pipeline(
        StandardScaler(),
        PCA(n_components=N_PCA, random_state=RANDOM_SEED),
    )

    X_pca = pca_pipe.fit_transform(X)
    np.save(OUT_DIR / "X_pca_three_dataset.npy", X_pca)

    print("Saved PCA embeddings.")

    if not HAS_SCVI:
        print("\nscVI not available. Skipping scVI.")
        print("Import error:")
        print(SCVI_IMPORT_ERROR)
        return

    print("\nPreparing counts for scVI...")
    counts = combined.layers["counts_like"]

    if sp.issparse(counts):
        counts = counts.tocsr().astype(np.float32)
        counts.data = np.round(counts.data)
        combined.layers["counts_scvi"] = counts
    else:
        counts = np.asarray(counts, dtype=np.float32)
        counts = np.round(counts)
        combined.layers["counts_scvi"] = counts

    print("Training scVI...")
    scvi.model.SCVI.setup_anndata(
        combined,
        layer="counts_scvi",
        batch_key="dataset_eval",
    )

    model = scvi.model.SCVI(
        combined,
        n_latent=N_LATENT,
        n_layers=2,
        gene_likelihood="nb",
    )

    model.train(
        max_epochs=SCVI_EPOCHS,
        accelerator="auto",
        devices="auto",
    )

    X_scvi = model.get_latent_representation()
    np.save(OUT_DIR / "X_scvi_three_dataset.npy", X_scvi)
    combined.obs.to_csv(OUT_DIR / "obs_scvi_three_dataset.csv")
    model.save(OUT_DIR / "scvi_three_dataset_model", overwrite=True)

    print("Saved scVI embeddings.")
    print(f"\nOutputs saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
