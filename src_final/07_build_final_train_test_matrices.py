from pathlib import Path
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp


ROOT = Path.cwd()

OUT_DIR = ROOT / "data" / "final_benchmark_matrices"
OUT_DIR.mkdir(parents=True, exist_ok=True)

REPORT_DIR = ROOT / "results_final6" / "final_matrices"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 0
rng = np.random.default_rng(RANDOM_SEED)

MAX_PER_CLASS_PER_TRAIN_DATASET = 12000

FILES = {
    "garrido": {
        "path": ROOT / "data" / "raw" / "cellxgene" / "ibd_garrido_trigo" / "manual_download" / "cellxgene_download.h5ad",
        "role": "train_dev",
    },
    "kong": {
        "path": ROOT / "data" / "processed" / "ibd_kong_2023_processed.h5ad",
        "role": "train_dev",
    },
    "smillie_epi": {
        "path": ROOT / "data" / "processed" / "ibd_smillie_uc_processed.h5ad",
        "role": "train_dev",
    },
    "martin": {
        "path": ROOT / "data" / "processed_final6" / "martin_locked_test_final.h5ad",
        "role": "locked_test",
    },
    "oliver_no_martin": {
        "path": ROOT / "data" / "processed_final6" / "oliver_no_martin_locked_test_final.h5ad",
        "role": "locked_test",
    },
}


def strip_ensembl_version(x):
    x = str(x).strip()
    if x.startswith("ENSG") and "." in x:
        return x.split(".")[0]
    return x


def looks_like_ensembl(x):
    return str(x).startswith("ENSG")


def disease_to_y(x):
    s = str(x).lower()
    if "normal" in s or "healthy" in s or "control" in s:
        return 0
    if "crohn" in s or "ulcerative" in s or "colitis" in s or "ibd" in s:
        return 1
    return np.nan


def infer_broad_cell_group(cell_type):
    s = str(cell_type).lower()

    epithelial_terms = [
        "epithelial", "colonocyte", "enterocyte", "goblet", "tuft", "paneth",
        "enteroendocrine", "crypt", "stem cell", "transit amplifying",
        "best4", "m cell", "foveolar", "mucous neck"
    ]
    lymphoid_terms = ["t cell", "nk cell", "b cell", "plasma", "lymphocyte"]
    myeloid_terms = ["myeloid", "macrophage", "monocyte", "dendritic", "neutrophil"]
    stromal_terms = ["stromal", "fibroblast", "endothelial", "pericyte", "smooth muscle"]

    if any(t in s for t in epithelial_terms):
        return "epithelial"
    if any(t in s for t in lymphoid_terms):
        return "lymphoid_plasma"
    if any(t in s for t in myeloid_terms):
        return "myeloid"
    if any(t in s for t in stromal_terms):
        return "stromal_endothelial"

    return "other"


def load_raw(dataset_id, info):
    path = info["path"]
    if not path.exists():
        raise FileNotFoundError(f"Missing {dataset_id}: {path}")
    a = ad.read_h5ad(path)
    print(f"Loaded {dataset_id}: {a.shape}")
    return a


def build_ensembl_to_symbol_map(raw_pieces):
    """
    Build map from Ensembl IDs to gene symbols using datasets that have both.
    This fixes Kong/Oliver/Martin/Garrido gene naming differences.
    """
    mapping = {}

    for dataset_id, a in raw_pieces.items():
        var = a.var.copy()
        var_names = [strip_ensembl_version(x) for x in a.var_names]

        symbol_col = None
        for c in ["gene_symbol", "gene_symbols", "feature_name"]:
            if c in var.columns:
                symbol_col = c
                break

        if symbol_col is None:
            continue

        symbols = var[symbol_col].astype(str).values

        for ens, sym in zip(var_names, symbols):
            ens = strip_ensembl_version(ens)
            sym = str(sym).strip()

            if not looks_like_ensembl(ens):
                continue
            if sym == "" or sym.lower() in ["nan", "none"]:
                continue
            if sym.startswith("ENSG"):
                continue

            if ens not in mapping:
                mapping[ens] = sym

    print(f"Built Ensembl→symbol map with {len(mapping):,} entries")
    return mapping


def standardize_gene_names(adata, dataset_id, ens_to_symbol):
    """
    Final feature names should be gene symbols when possible.
    If var_names are Ensembl IDs, map them to symbols using ens_to_symbol.
    If the dataset already has gene symbols, use those.
    """
    var = adata.var.copy()
    original = [str(x) for x in adata.var_names]

    names = None
    source = None

    # Prefer explicit symbol columns when they exist.
    for c in ["gene_symbol", "gene_symbols", "feature_name"]:
        if c in var.columns:
            candidate = var[c].astype(str).values
            n_good = sum(
                (str(x).strip() != "")
                and (str(x).lower() not in ["nan", "none"])
                and (not str(x).startswith("ENSG"))
                for x in candidate
            )

            if n_good > 1000:
                names = list(candidate)
                source = c
                break

    # Otherwise map Ensembl var_names to symbols.
    if names is None:
        mapped = []
        n_mapped = 0

        for x in original:
            x_clean = strip_ensembl_version(x)
            if x_clean in ens_to_symbol:
                mapped.append(ens_to_symbol[x_clean])
                n_mapped += 1
            else:
                mapped.append(x_clean)

        names = mapped
        source = f"var_names_mapped_ensembl_to_symbol_n={n_mapped}"

    cleaned = []
    for i, g in enumerate(names):
        g = str(g).strip()
        if g == "" or g.lower() in ["nan", "none"]:
            g = f"unknown_gene_{i}"

        # If still Ensembl with version, strip version.
        g = strip_ensembl_version(g)

        cleaned.append(g)

    adata.var_names = pd.Index(cleaned).astype(str)
    adata.var_names_make_unique()

    n_ensg_remaining = sum(str(x).startswith("ENSG") for x in adata.var_names[: min(5000, adata.n_vars)])
    print(f"{dataset_id}: gene name source = {source}")
    print(f"{dataset_id}: first 12 standardized genes = {list(adata.var_names[:12])}")
    print(f"{dataset_id}: ENSG remaining in first genes = {n_ensg_remaining}")

    return adata


def ensure_obs(adata, dataset_id, role):
    obs = adata.obs.copy()

    if "y_ibd" in obs.columns:
        y = pd.to_numeric(obs["y_ibd"], errors="coerce")
    elif "disease" in obs.columns:
        y = obs["disease"].map(disease_to_y)
    elif "disease_label_raw" in obs.columns:
        y = obs["disease_label_raw"].map(disease_to_y)
    else:
        raise ValueError(f"{dataset_id}: cannot infer y_ibd")

    adata.obs["y_ibd"] = y.astype(float)

    if "donor_label" in obs.columns:
        donor = obs["donor_label"].astype(str)
    elif "donor_id" in obs.columns:
        donor = obs["donor_id"].astype(str)
    elif "donorID_unified" in obs.columns:
        donor = obs["donorID_unified"].astype(str)
    elif "Subject" in obs.columns:
        donor = obs["Subject"].astype(str)
    else:
        donor = pd.Series(["unknown"] * adata.n_obs, index=adata.obs_names)

    adata.obs["donor_label"] = donor.values

    if "cell_type_label" in obs.columns:
        ct = obs["cell_type_label"].astype(str)
    elif "cell_type" in obs.columns:
        ct = obs["cell_type"].astype(str)
    elif "Celltype" in obs.columns:
        ct = obs["Celltype"].astype(str)
    elif "CellType" in obs.columns:
        ct = obs["CellType"].astype(str)
    else:
        ct = pd.Series(["unknown"] * adata.n_obs, index=adata.obs_names)

    adata.obs["cell_type_label"] = ct.values

    if "broad_cell_group" in obs.columns:
        broad = obs["broad_cell_group"].astype(str)
    else:
        broad = ct.map(infer_broad_cell_group).astype(str)

    adata.obs["broad_cell_group"] = broad.values

    adata.obs["dataset_eval"] = dataset_id
    adata.obs["dataset_id"] = dataset_id
    adata.obs["role"] = role

    if "tissue_label" not in adata.obs.columns:
        if "tissue" in obs.columns:
            adata.obs["tissue_label"] = obs["tissue"].astype(str)
        else:
            adata.obs["tissue_label"] = "unknown"

    if "disease_label_raw" not in adata.obs.columns:
        if "disease" in obs.columns:
            adata.obs["disease_label_raw"] = obs["disease"].astype(str)
        else:
            adata.obs["disease_label_raw"] = adata.obs["y_ibd"].astype(str)

    if "n_counts" not in adata.obs.columns:
        if sp.issparse(adata.X):
            adata.obs["n_counts"] = np.asarray(adata.X.sum(axis=1)).ravel()
        else:
            adata.obs["n_counts"] = np.asarray(adata.X.sum(axis=1)).ravel()

    keep = adata.obs["y_ibd"].notna().values
    adata = adata[keep].copy()
    adata.obs["y_ibd"] = adata.obs["y_ibd"].astype(int)

    return adata


def downsample_train_dataset(adata, dataset_id):
    chosen = []

    for label in [0, 1]:
        idx = np.where(adata.obs["y_ibd"].values == label)[0]
        if len(idx) == 0:
            print(f"WARNING {dataset_id}: no class {label}")
            continue

        n_take = min(len(idx), MAX_PER_CLASS_PER_TRAIN_DATASET)
        chosen.extend(rng.choice(idx, size=n_take, replace=False).tolist())

    chosen = np.array(sorted(chosen), dtype=int)
    return adata[chosen].copy()


def clean_obs_for_write(adata):
    keep_cols = [
        "dataset_eval",
        "dataset_id",
        "role",
        "y_ibd",
        "disease_label_raw",
        "cell_type_label",
        "broad_cell_group",
        "donor_label",
        "tissue_label",
        "n_counts",
    ]

    for extra in [
        "subdataset",
        "locked_test_dataset",
        "derived_from_oliver",
        "martin_gse134809_removed",
        "sample_label",
        "assay_label",
        "source_study",
        "source_id",
    ]:
        if extra in adata.obs.columns:
            keep_cols.append(extra)

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
    adata.var_names_make_unique()

    return adata


def load_and_standardize(dataset_id, info, raw_pieces, ens_to_symbol):
    print("\n" + "=" * 120)
    print(f"Standardizing {dataset_id}")

    adata = raw_pieces[dataset_id].copy()
    adata = standardize_gene_names(adata, dataset_id, ens_to_symbol)
    adata = ensure_obs(adata, dataset_id, info["role"])

    print("After obs standardization:", adata.shape)
    print("y_ibd counts:")
    print(adata.obs["y_ibd"].value_counts())
    print("broad groups:")
    print(adata.obs["broad_cell_group"].value_counts().head(20))
    print("donors:", adata.obs["donor_label"].nunique())

    if info["role"] == "train_dev":
        adata = downsample_train_dataset(adata, dataset_id)
        print("After train downsample:", adata.shape)
        print(adata.obs["y_ibd"].value_counts())

    adata = clean_obs_for_write(adata)
    return adata


def subset_to_common_genes(pieces):
    gene_sets = {}
    for name, a in pieces.items():
        genes = set(a.var_names.astype(str))
        gene_sets[name] = genes
        print(f"{name}: {len(genes):,} genes after standardization")

    common = set.intersection(*gene_sets.values())
    common = sorted(common)

    print("\n" + "=" * 120)
    print(f"Common genes across all datasets: {len(common):,}")

    if len(common) < 5000:
        print("ERROR: fewer than 5,000 common genes.")
        print("Pairwise overlaps:")
        names = list(gene_sets.keys())
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = names[i], names[j]
                print(f"{a} vs {b}: {len(gene_sets[a].intersection(gene_sets[b])):,}")
        raise ValueError("Too few common genes after fixed gene standardization.")

    subset = {}
    for name, a in pieces.items():
        subset[name] = a[:, common].copy()
        print(f"{name} common-gene shape: {subset[name].shape}")

    return subset, common


def save_group(name, pieces):
    print("\n" + "=" * 120)
    print(f"Saving group: {name}")

    combined = ad.concat(
        pieces,
        join="inner",
        index_unique="-",
    )

    combined.var_names_make_unique()
    combined = clean_obs_for_write(combined)

    if combined.n_vars == 0:
        raise ValueError(f"{name} has 0 genes. Stop.")

    out = OUT_DIR / f"{name}.h5ad"
    obs_out = OUT_DIR / f"{name}_obs.csv"

    combined.write_h5ad(out, compression="gzip")
    combined.obs.to_csv(obs_out)

    print("Saved:")
    print(out)
    print(obs_out)
    print("Shape:", combined.shape)
    print("dataset counts:")
    print(combined.obs["dataset_eval"].value_counts())
    print("y counts:")
    print(combined.obs["y_ibd"].value_counts())
    print("broad groups:")
    print(combined.obs["broad_cell_group"].value_counts())

    return combined


def main():
    print("Building final benchmark matrices with FIXED gene-name harmonization...")

    raw_pieces = {}
    for dataset_id, info in FILES.items():
        raw_pieces[dataset_id] = load_raw(dataset_id, info)

    ens_to_symbol = build_ensembl_to_symbol_map(raw_pieces)

    pieces = {}
    for dataset_id, info in FILES.items():
        pieces[dataset_id] = load_and_standardize(dataset_id, info, raw_pieces, ens_to_symbol)

    pieces, common_genes = subset_to_common_genes(pieces)

    pd.DataFrame({"gene": common_genes}).to_csv(
        OUT_DIR / "final_common_genes.csv",
        index=False,
    )

    train_pieces = [
        pieces["garrido"],
        pieces["kong"],
        pieces["smillie_epi"],
    ]

    martin_piece = pieces["martin"]
    oliver_piece = pieces["oliver_no_martin"]

    train_all = save_group("train_dev_all_cells_common", train_pieces)
    martin_all = save_group("locked_test_martin_all_cells_common", [martin_piece])
    oliver_all = save_group("locked_test_oliver_no_martin_all_cells_common", [oliver_piece])

    train_epi_pieces = [
        a[a.obs["broad_cell_group"].astype(str).values == "epithelial"].copy()
        for a in train_pieces
    ]
    martin_epi_piece = martin_piece[
        martin_piece.obs["broad_cell_group"].astype(str).values == "epithelial"
    ].copy()
    oliver_epi_piece = oliver_piece[
        oliver_piece.obs["broad_cell_group"].astype(str).values == "epithelial"
    ].copy()

    train_epi = save_group("train_dev_epithelial_common", train_epi_pieces)
    martin_epi = save_group("locked_test_martin_epithelial_common", [martin_epi_piece])
    oliver_epi = save_group("locked_test_oliver_no_martin_epithelial_common", [oliver_epi_piece])

    summary_rows = []

    for name, a in [
        ("train_dev_all_cells_common", train_all),
        ("locked_test_martin_all_cells_common", martin_all),
        ("locked_test_oliver_no_martin_all_cells_common", oliver_all),
        ("train_dev_epithelial_common", train_epi),
        ("locked_test_martin_epithelial_common", martin_epi),
        ("locked_test_oliver_no_martin_epithelial_common", oliver_epi),
    ]:
        summary_rows.append({
            "matrix": name,
            "n_cells": a.n_obs,
            "n_genes": a.n_vars,
            "n_class_0": int((a.obs["y_ibd"].values == 0).sum()),
            "n_class_1": int((a.obs["y_ibd"].values == 1).sum()),
            "datasets": ";".join(sorted(a.obs["dataset_eval"].astype(str).unique())),
            "broad_groups": ";".join(sorted(a.obs["broad_cell_group"].astype(str).unique())),
        })

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(REPORT_DIR / "final_matrix_summary.csv", index=False)

    print("\n" + "=" * 120)
    print("Final matrix summary:")
    print(summary.to_string(index=False))
    print("\nSaved summary:")
    print(REPORT_DIR / "final_matrix_summary.csv")


if __name__ == "__main__":
    main()