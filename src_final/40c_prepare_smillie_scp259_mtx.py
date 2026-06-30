from pathlib import Path
import gzip
import shutil
import tempfile

import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp
from scipy.io import mmread


ROOT = Path.cwd()

RAW_DIR = ROOT / "data" / "raw" / "smillie_scp259"
OUT_DIR = ROOT / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_FILE = OUT_DIR / "ibd_smillie_uc_processed.h5ad"
OUT_OBS = OUT_DIR / "ibd_smillie_uc_obs.csv"

RANDOM_SEED = 0
MAX_CELLS_TOTAL = 40000

rng = np.random.default_rng(RANDOM_SEED)


# EPI ONLY.
# Your metadata file matches epithelial barcodes, but not Fib/Imm barcodes.
# So for the third-dataset benchmark, Smillie should be used as epithelial-only.
COMPARTMENTS = {
    "Epi": {
        "label": "epithelial",
        "genes": "Epi.genes.tsv",
        "barcodes": "Epi.barcodes2.tsv",
        "matrix": "gene_sorted-Epi.matrix.mtx",
    },
}


def find_existing(base_name):
    """
    Find file even if Windows/browser adds .txt or .gz.
    Examples handled:
    all.meta2.txt
    all.meta2.txt.txt
    all.meta2.txt.gz
    gene_sorted-Epi.matrix.mtx
    gene_sorted-Epi.matrix.mtx.gz
    """
    candidates = [
        RAW_DIR / base_name,
        RAW_DIR / f"{base_name}.gz",
        RAW_DIR / f"{base_name}.txt",
        RAW_DIR / f"{base_name}.txt.gz",
    ]

    for p in candidates:
        if p.exists():
            return p

    matches = sorted(RAW_DIR.glob(base_name + "*"))
    if matches:
        return matches[0]

    raise FileNotFoundError(
        f"Could not find {base_name} or compressed/renamed version in:\n{RAW_DIR}"
    )


def read_table_auto(path, **kwargs):
    path = Path(path)

    if path.suffix == ".gz":
        return pd.read_csv(
            path,
            sep="\t",
            compression="gzip",
            low_memory=False,
            **kwargs,
        )

    return pd.read_csv(
        path,
        sep="\t",
        low_memory=False,
        **kwargs,
    )


def read_lines_auto(path):
    path = Path(path)

    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return [line.rstrip("\n") for line in f]

    with open(path, "r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def mmread_auto(path):
    path = Path(path)

    if path.suffix != ".gz":
        return mmread(str(path))

    with tempfile.NamedTemporaryFile(suffix=".mtx", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        with gzip.open(path, "rb") as f_in:
            with open(tmp_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)

        return mmread(str(tmp_path))
    finally:
        tmp_path.unlink(missing_ok=True)


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


def load_genes(path):
    df = read_table_auto(path, header=None)

    if df.shape[1] == 1:
        genes = df.iloc[:, 0].astype(str).values
    else:
        genes = df.iloc[:, 0].astype(str).values

    cleaned = []
    for i, g in enumerate(genes):
        g = str(g).strip()
        if g in ["", "nan", "None"]:
            g = f"gene_{i}"
        cleaned.append(g)

    return make_unique_names(cleaned)


def load_barcodes(path):
    lines = read_lines_auto(path)
    barcodes = []

    for line in lines:
        if not line.strip():
            continue

        # Keep first tab-separated field as barcode.
        barcodes.append(line.split("\t")[0].strip())

    return np.array(barcodes, dtype=str)


def disease_to_binary(x):
    """
    For Smillie SCP259:
    Healthy = control = 0
    Inflamed + Non-inflamed = UC patient tissue = 1

    Non-inflamed still comes from UC patients, so it belongs to disease/UC.
    """
    if pd.isna(x):
        return np.nan

    s = str(x).strip().lower()

    if s in ["", "nan", "none", "group"]:
        return np.nan

    if s in ["0", "0.0", "false", "no"]:
        return 0

    if "healthy" in s or "normal" in s or "control" in s:
        return 0

    if (
        s == "uc"
        or "ulcerative" in s
        or "colitis" in s
        or "ibd" in s
        or "crohn" in s
        or "inflamed" in s
        or "non-inflamed" in s
        or "noninflamed" in s
        or "inactive" in s
        or "active" in s
    ):
        return 1

    return np.nan


def load_metadata():
    meta_path = find_existing("all.meta2.txt")

    print(f"Loading metadata: {meta_path}")

    meta = read_table_auto(meta_path, index_col=0)
    meta.index = meta.index.astype(str)
    meta["__meta_index__"] = meta.index.astype(str)

    print("Metadata shape:", meta.shape)
    print("Metadata columns:")
    print(list(meta.columns))

    print("\nCandidate label columns:")
    for c in ["Health", "health", "disease", "Disease", "condition", "Condition", "diagnosis", "Diagnosis", "Sample", "sample"]:
        if c in meta.columns:
            print(f"\n--- {c} ---")
            print(meta[c].astype(str).value_counts(dropna=False).head(40))

    return meta


def find_best_metadata_key(meta, barcodes):
    barcode_set = set(map(str, barcodes))

    candidates = {
        "__meta_index__": meta["__meta_index__"].astype(str),
    }

    for col in meta.columns:
        try:
            vals = meta[col].astype(str)
            candidates[col] = vals
        except Exception:
            pass

    best_name = None
    best_overlap = -1

    for name, vals in candidates.items():
        overlap = len(set(vals).intersection(barcode_set))

        if overlap > best_overlap:
            best_overlap = overlap
            best_name = name

    print(f"Best metadata key: {best_name}, exact overlap: {best_overlap}/{len(barcodes)}")

    if best_overlap < 100:
        print("\nWARNING: Low metadata/barcode overlap.")
        print("First 10 barcodes:")
        print(list(barcodes[:10]))
        print("First 10 metadata index values:")
        print(list(meta.index[:10]))

    return best_name, best_overlap


def pick_label_col(meta):
    preferred = [
        "Health",
        "health",
        "disease",
        "Disease",
        "condition",
        "Condition",
        "diagnosis",
        "Diagnosis",
    ]

    for col in preferred:
        if col in meta.columns:
            y = meta[col].map(disease_to_binary)

            if y.notna().sum() > 0 and len(set(y.dropna().astype(int))) >= 2:
                print(f"Using disease label column: {col}")
                print(meta[col].astype(str).value_counts(dropna=False).head(40))
                return col

    # Fallback: scan every column.
    for col in meta.columns:
        try:
            y = meta[col].map(disease_to_binary)

            if y.notna().sum() > 0 and len(set(y.dropna().astype(int))) >= 2:
                print(f"Using fallback disease label column: {col}")
                print(meta[col].astype(str).value_counts(dropna=False).head(40))
                return col
        except Exception:
            pass

    raise ValueError(
        "Could not find a metadata column with both healthy/control and UC/inflamed labels."
    )


def pick_col(df, options, fallback=None):
    for col in options:
        if col in df.columns:
            return col
    return fallback


def load_compartment(compartment_key, config, meta, label_col):
    print("\n" + "=" * 100)
    print(f"Loading compartment: {compartment_key}")

    genes_path = find_existing(config["genes"])
    barcodes_path = find_existing(config["barcodes"])
    matrix_path = find_existing(config["matrix"])

    print("Genes:", genes_path)
    print("Barcodes:", barcodes_path)
    print("Matrix:", matrix_path)

    genes = load_genes(genes_path)
    barcodes = load_barcodes(barcodes_path)

    print(f"n_genes from file: {len(genes):,}")
    print(f"n_barcodes from file: {len(barcodes):,}")

    M = mmread_auto(matrix_path)

    if not sp.issparse(M):
        M = sp.csr_matrix(M)
    else:
        M = M.tocsr()

    print("Raw matrix shape:", M.shape)

    # Convert to cells x genes.
    if M.shape[0] == len(genes) and M.shape[1] == len(barcodes):
        X = M.T.tocsr()
        print("Matrix orientation: genes x cells; transposed to cells x genes.")
    elif M.shape[0] == len(barcodes) and M.shape[1] == len(genes):
        X = M.tocsr()
        print("Matrix orientation: cells x genes.")
    else:
        raise ValueError(
            f"Matrix shape {M.shape} does not match genes={len(genes)} and barcodes={len(barcodes)}"
        )

    key_col, overlap = find_best_metadata_key(meta, barcodes)

    if overlap == 0:
        raise ValueError(
            f"No metadata overlap for {compartment_key}. "
            f"This script is intended for compartments whose metadata barcodes match."
        )

    meta_keyed = meta.copy()
    meta_keyed["__join_key__"] = meta_keyed[key_col].astype(str)
    meta_keyed = meta_keyed.drop_duplicates("__join_key__").set_index("__join_key__")

    obs = pd.DataFrame(index=barcodes)
    obs["barcode"] = barcodes
    obs["source_compartment"] = compartment_key
    obs["broad_cell_group_source"] = config["label"]

    joined = obs.join(meta_keyed, how="left")

    matched = joined[label_col].notna().sum()
    print(f"Metadata matched cells: {matched:,}/{len(joined):,}")

    y = joined[label_col].map(disease_to_binary)
    keep = y.notna().values

    print("Disease counts before sampling:")
    print(y.dropna().astype(int).value_counts())

    if keep.sum() == 0:
        raise ValueError(f"No usable labels for {compartment_key}")

    idx_all = np.where(keep)[0]
    y_keep = y.iloc[idx_all].astype(int).values

    chosen = []

    # Balanced sample.
    for label in [0, 1]:
        idx_label = idx_all[y_keep == label]

        if len(idx_label) == 0:
            print(f"WARNING: no label {label} in {compartment_key}")
            continue

        n_take = min(len(idx_label), MAX_CELLS_TOTAL // 2)
        chosen.extend(rng.choice(idx_label, size=n_take, replace=False).tolist())

    chosen = np.array(sorted(chosen), dtype=int)

    X_sub = X[chosen, :].tocsr()
    obs_sub = joined.iloc[chosen].copy()

    obs_sub["dataset_eval"] = "smillie"
    obs_sub["dataset_id"] = "smillie_scp259"
    obs_sub["disease_label_raw"] = obs_sub[label_col].astype(str)
    obs_sub["y_ibd"] = obs_sub[label_col].map(disease_to_binary).astype(int)

    celltype_col = pick_col(
        obs_sub,
        ["Cluster", "CellType", "cell_type", "celltype"],
        fallback=None,
    )
    subject_col = pick_col(
        obs_sub,
        ["Subject", "subject", "donor_id", "donor", "patient_id"],
        fallback=None,
    )
    location_col = pick_col(
        obs_sub,
        ["Location", "location", "tissue", "Tissue"],
        fallback=None,
    )
    sample_col = pick_col(
        obs_sub,
        ["Sample", "sample"],
        fallback=None,
    )

    if celltype_col is not None:
        obs_sub["cell_type_label"] = obs_sub[celltype_col].astype(str)
    else:
        obs_sub["cell_type_label"] = config["label"]

    if subject_col is not None:
        obs_sub["donor_label"] = obs_sub[subject_col].astype(str)
    elif sample_col is not None:
        obs_sub["donor_label"] = obs_sub[sample_col].astype(str)
    else:
        obs_sub["donor_label"] = "unknown"

    if location_col is not None:
        obs_sub["tissue_label"] = obs_sub[location_col].astype(str)
    else:
        obs_sub["tissue_label"] = "colon"

    # Epi-only Smillie should be epithelial.
    obs_sub["broad_cell_group"] = "epithelial"

    # Build AnnData.
    a = ad.AnnData(
        X=X_sub,
        obs=obs_sub,
        var=pd.DataFrame(index=genes),
    )

    a.var_names_make_unique()

    print("Selected shape:", a.shape)
    print("Selected disease counts:")
    print(a.obs["y_ibd"].value_counts())

    return a


def clean_before_write(combined):
    """
    Remove messy original metadata columns that break h5ad writing.
    The crash you saw was caused by mixed-type metadata like nGene.
    """
    keep_obs_cols = [
        "dataset_eval",
        "dataset_id",
        "source_compartment",
        "broad_cell_group_source",
        "disease_label_raw",
        "y_ibd",
        "cell_type_label",
        "donor_label",
        "tissue_label",
        "broad_cell_group",
        "barcode",
    ]

    keep_obs_cols = [c for c in keep_obs_cols if c in combined.obs.columns]
    combined.obs = combined.obs[keep_obs_cols].copy()

    for col in combined.obs.columns:
        if col == "y_ibd":
            combined.obs[col] = combined.obs[col].astype(int)
        else:
            combined.obs[col] = combined.obs[col].astype(str).fillna("unknown")

    combined.obs_names = combined.obs_names.astype(str)
    combined.var_names = combined.var_names.astype(str)
    combined.var_names_make_unique()

    return combined


def main():
    print("Checking required Epi files...")

    required_names = [
        "all.meta2.txt",
        "Epi.genes.tsv",
        "Epi.barcodes2.tsv",
        "gene_sorted-Epi.matrix.mtx",
    ]

    for name in required_names:
        path = find_existing(name)
        print("FOUND:", path.name)

    meta = load_metadata()
    label_col = pick_label_col(meta)

    pieces = []

    for key, config in COMPARTMENTS.items():
        piece = load_compartment(key, config, meta, label_col)
        pieces.append(piece)

    print("\n" + "=" * 100)
    print("Concatenating Smillie epithelial compartment...")

    combined = ad.concat(
        pieces,
        join="outer",
        fill_value=0,
        index_unique="-",
    )

    combined.var_names_make_unique()

    print("Combined Smillie shape:", combined.shape)

    print("\nDisease counts:")
    print(combined.obs["y_ibd"].value_counts())

    print("\nDisease raw labels:")
    print(combined.obs["disease_label_raw"].value_counts().head(30))

    print("\nBroad cell groups:")
    print(combined.obs["broad_cell_group"].value_counts())

    print("\nDonor count:")
    print(combined.obs["donor_label"].nunique())

    if combined.obs["y_ibd"].nunique() < 2:
        raise ValueError("Smillie still has only one class. Do not use this output.")

    combined = clean_before_write(combined)

    print("\nClean obs columns before saving:")
    print(list(combined.obs.columns))
    print(combined.obs.dtypes)

    print("\nWriting processed Smillie h5ad...")
    combined.write_h5ad(OUT_FILE, compression="gzip")
    combined.obs.to_csv(OUT_OBS)

    print("\nSaved:")
    print(OUT_FILE)
    print(OUT_OBS)

    print("\nFinal verification:")
    print("Shape:", combined.shape)
    print("Disease counts:")
    print(combined.obs["y_ibd"].value_counts())
    print("Broad groups:")
    print(combined.obs["broad_cell_group"].value_counts())
    print("Donors:", combined.obs["donor_label"].nunique())


if __name__ == "__main__":
    main()
