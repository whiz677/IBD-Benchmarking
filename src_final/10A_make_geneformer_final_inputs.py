from pathlib import Path
import sys
import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

ROOT = Path.cwd()
IN_DIR = ROOT / "data" / "final_benchmark_matrices"
OUT_DIR = ROOT / "data" / "geneformer_final_inputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FILES = {
    "train_dev_all_cells_common": IN_DIR / "train_dev_all_cells_common.h5ad",
    "locked_test_martin_all_cells_common": IN_DIR / "locked_test_martin_all_cells_common.h5ad",
    "locked_test_oliver_no_martin_all_cells_common": IN_DIR / "locked_test_oliver_no_martin_all_cells_common.h5ad",
}

def looks_ensembl(x):
    x = str(x)
    return x.startswith("ENSG") or x.startswith("ENSMUSG")

def n_counts(X):
    if sparse.issparse(X):
        return np.asarray(X.sum(axis=1)).ravel()
    return np.asarray(X).sum(axis=1)

def find_ensembl_column(a):
    candidates = [
        "ensembl_id",
        "ensembl",
        "gene_id",
        "gene_ids",
        "feature_id",
        "ensembl_gene_id",
        "gene_stable_id",
    ]

    for c in candidates:
        if c in a.var.columns:
            s = a.var[c].astype(str)
            if s.map(looks_ensembl).sum() > 100:
                return c

    idx = pd.Series(a.var_names.astype(str), index=a.var_names)
    if idx.map(looks_ensembl).sum() > 100:
        return "__var_names__"

    return None

print("Checking input files...")
for name, path in FILES.items():
    if not path.exists():
        print("MISSING:", path)
        sys.exit(1)
    print(name, path)

print("\nMaking Geneformer-ready files...")

for name, path in FILES.items():
    print("\n" + "=" * 90)
    print("Reading:", path)
    a = ad.read_h5ad(path)

    print("shape:", a.shape)
    print("obs columns:", list(a.obs.columns)[:30])
    print("var columns:", list(a.var.columns)[:30])
    print("first var names:", list(a.var_names[:10]))

    col = find_ensembl_column(a)

    if col is None:
        print("\nERROR: Could not find Ensembl IDs.")
        print("Geneformer needs Ensembl gene IDs like ENSG00000141510.")
        print("Your first var names are:")
        print(list(a.var_names[:30]))
        print("Your var columns are:")
        print(list(a.var.columns))
        sys.exit(1)

    if col == "__var_names__":
        a.var["ensembl_id"] = a.var_names.astype(str)
    else:
        a.var["ensembl_id"] = a.var[col].astype(str)

    keep = a.var["ensembl_id"].astype(str).map(looks_ensembl).values
    a = a[:, keep].copy()

    keep_unique = ~pd.Index(a.var["ensembl_id"]).duplicated()
    a = a[:, keep_unique].copy()

    if "n_counts" not in a.obs.columns:
        a.obs["n_counts"] = n_counts(a.X)

    out = OUT_DIR / f"{name}_geneformer_ready.h5ad"
    a.write_h5ad(out)
    print("WROTE:", out)
    print("final shape:", a.shape)

print("\nDONE 10A. Files:")
for p in sorted(OUT_DIR.glob("*.h5ad")):
    print(" ", p)
