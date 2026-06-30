from pathlib import Path
import numpy as np
import pandas as pd
import anndata as ad

ROOT = Path.cwd()
OUT = ROOT / "results_final6" / "proper_no_leak_benchmark"
OBS_OUT = OUT / "obs_for_embeddings"
GF_EPI_OUT = ROOT / "embeddings_final6" / "geneformer_epithelial"

OUT.mkdir(parents=True, exist_ok=True)
OBS_OUT.mkdir(parents=True, exist_ok=True)
GF_EPI_OUT.mkdir(parents=True, exist_ok=True)

H5ADS = {
    ("all_cells", "train_dev"): ROOT / "data/final_benchmark_matrices/train_dev_all_cells_common.h5ad",
    ("all_cells", "martin"): ROOT / "data/final_benchmark_matrices/locked_test_martin_all_cells_common.h5ad",
    ("all_cells", "oliver_no_martin"): ROOT / "data/final_benchmark_matrices/locked_test_oliver_no_martin_all_cells_common.h5ad",
    ("epithelial", "train_dev"): ROOT / "data/final_benchmark_matrices/train_dev_epithelial_common.h5ad",
    ("epithelial", "martin"): ROOT / "data/final_benchmark_matrices/locked_test_martin_epithelial_common.h5ad",
    ("epithelial", "oliver_no_martin"): ROOT / "data/final_benchmark_matrices/locked_test_oliver_no_martin_epithelial_common.h5ad",
}

obs_paths = {}

print("\nWriting obs files from final benchmark matrices...")
for key, h5ad_path in H5ADS.items():
    group, split = key
    if not h5ad_path.exists():
        print("MISSING H5AD:", h5ad_path)
        continue

    a = ad.read_h5ad(h5ad_path, backed="r")
    obs = a.obs.copy()
    a.file.close()

    out = OBS_OUT / f"{group}_{split}_obs.csv"
    obs.to_csv(out, index=False)
    obs_paths[key] = out

    print(f"{group:10s} {split:18s}", obs.shape, "->", out)

print("\nCreating Geneformer epithelial subset embeddings from all-cells Geneformer embeddings...")

GF_ALL = {
    "train_dev": (
        ROOT / "embeddings_final6/geneformer/train_dev_all_cells_common_geneformer_embeddings.npy",
        ROOT / "embeddings_final6/geneformer/train_dev_all_cells_common_geneformer_obs.csv",
    ),
    "martin": (
        ROOT / "embeddings_final6/geneformer/locked_test_martin_all_cells_common_geneformer_embeddings.npy",
        ROOT / "embeddings_final6/geneformer/locked_test_martin_all_cells_common_geneformer_obs.csv",
    ),
    "oliver_no_martin": (
        ROOT / "embeddings_final6/geneformer/locked_test_oliver_no_martin_all_cells_common_geneformer_embeddings.npy",
        ROOT / "embeddings_final6/geneformer/locked_test_oliver_no_martin_all_cells_common_geneformer_obs.csv",
    ),
}

gf_epi_files = {}

for split, (emb_path, obs_path) in GF_ALL.items():
    if not emb_path.exists() or not obs_path.exists():
        print("Missing Geneformer all-cells file for", split)
        continue

    X = np.load(emb_path)
    obs = pd.read_csv(obs_path)

    for c in ["Unnamed: 0", "index"]:
        if c in obs.columns:
            obs = obs.drop(columns=[c])

    if X.shape[0] != len(obs):
        raise ValueError(f"Geneformer row mismatch {split}: {X.shape} vs {obs.shape}")

    mask = obs["broad_cell_group"].astype(str).str.lower().eq("epithelial").values
    X_epi = X[mask]
    obs_epi = obs.loc[mask].reset_index(drop=True)

    out_emb = GF_EPI_OUT / f"epithelial_geneformer_{split}_embeddings.npy"
    out_obs = GF_EPI_OUT / f"epithelial_geneformer_{split}_obs.csv"

    np.save(out_emb, X_epi)
    obs_epi.to_csv(out_obs, index=False)

    gf_epi_files[split] = (out_emb, out_obs)
    print(split, X_epi.shape, obs_epi.shape, "->", out_emb)

registry = []

def add(model, group, split, emb, obs):
    registry.append({
        "model": model,
        "group": group,
        "split": split,
        "embedding_file": str(emb),
        "obs_file": str(obs),
    })

# PCA
add("PCA_50", "all_cells", "train_dev",
    ROOT / "embeddings_final6/pca_scvi/all_cells_PCA_train.npy",
    obs_paths[("all_cells", "train_dev")])
add("PCA_50", "all_cells", "martin",
    ROOT / "embeddings_final6/pca_scvi/all_cells_PCA_test_martin.npy",
    obs_paths[("all_cells", "martin")])
add("PCA_50", "all_cells", "oliver_no_martin",
    ROOT / "embeddings_final6/pca_scvi/all_cells_PCA_test_oliver_no_martin.npy",
    obs_paths[("all_cells", "oliver_no_martin")])

add("PCA_50", "epithelial", "train_dev",
    ROOT / "embeddings_final6/pca_scvi/epithelial_PCA_train.npy",
    obs_paths[("epithelial", "train_dev")])
add("PCA_50", "epithelial", "martin",
    ROOT / "embeddings_final6/pca_scvi/epithelial_PCA_test_martin.npy",
    obs_paths[("epithelial", "martin")])
add("PCA_50", "epithelial", "oliver_no_martin",
    ROOT / "embeddings_final6/pca_scvi/epithelial_PCA_test_oliver_no_martin.npy",
    obs_paths[("epithelial", "oliver_no_martin")])

# scVI
add("scVI_30", "all_cells", "train_dev",
    ROOT / "embeddings_final6/pca_scvi/all_cells_scVI_train.npy",
    obs_paths[("all_cells", "train_dev")])
add("scVI_30", "all_cells", "martin",
    ROOT / "embeddings_final6/pca_scvi/all_cells_scVI_test_martin.npy",
    obs_paths[("all_cells", "martin")])
add("scVI_30", "all_cells", "oliver_no_martin",
    ROOT / "embeddings_final6/pca_scvi/all_cells_scVI_test_oliver_no_martin.npy",
    obs_paths[("all_cells", "oliver_no_martin")])

add("scVI_30", "epithelial", "train_dev",
    ROOT / "embeddings_final6/pca_scvi/epithelial_scVI_train.npy",
    obs_paths[("epithelial", "train_dev")])
add("scVI_30", "epithelial", "martin",
    ROOT / "embeddings_final6/pca_scvi/epithelial_scVI_test_martin.npy",
    obs_paths[("epithelial", "martin")])
add("scVI_30", "epithelial", "oliver_no_martin",
    ROOT / "embeddings_final6/pca_scvi/epithelial_scVI_test_oliver_no_martin.npy",
    obs_paths[("epithelial", "oliver_no_martin")])

# scGPT
add("scGPT_frozen", "all_cells", "train_dev",
    ROOT / "embeddings_final6/scgpt/all_cells_scGPT_train.npy",
    obs_paths[("all_cells", "train_dev")])
add("scGPT_frozen", "all_cells", "martin",
    ROOT / "embeddings_final6/scgpt/all_cells_scGPT_test_martin.npy",
    obs_paths[("all_cells", "martin")])
add("scGPT_frozen", "all_cells", "oliver_no_martin",
    ROOT / "embeddings_final6/scgpt/all_cells_scGPT_test_oliver_no_martin.npy",
    obs_paths[("all_cells", "oliver_no_martin")])

add("scGPT_frozen", "epithelial", "train_dev",
    ROOT / "embeddings_final6/scgpt/epithelial_scGPT_train.npy",
    obs_paths[("epithelial", "train_dev")])
add("scGPT_frozen", "epithelial", "martin",
    ROOT / "embeddings_final6/scgpt/epithelial_scGPT_test_martin.npy",
    obs_paths[("epithelial", "martin")])
add("scGPT_frozen", "epithelial", "oliver_no_martin",
    ROOT / "embeddings_final6/scgpt/epithelial_scGPT_test_oliver_no_martin.npy",
    obs_paths[("epithelial", "oliver_no_martin")])

# Geneformer all-cells
add("Geneformer_frozen", "all_cells", "train_dev",
    ROOT / "embeddings_final6/geneformer/train_dev_all_cells_common_geneformer_embeddings.npy",
    ROOT / "embeddings_final6/geneformer/train_dev_all_cells_common_geneformer_obs.csv")
add("Geneformer_frozen", "all_cells", "martin",
    ROOT / "embeddings_final6/geneformer/locked_test_martin_all_cells_common_geneformer_embeddings.npy",
    ROOT / "embeddings_final6/geneformer/locked_test_martin_all_cells_common_geneformer_obs.csv")
add("Geneformer_frozen", "all_cells", "oliver_no_martin",
    ROOT / "embeddings_final6/geneformer/locked_test_oliver_no_martin_all_cells_common_geneformer_embeddings.npy",
    ROOT / "embeddings_final6/geneformer/locked_test_oliver_no_martin_all_cells_common_geneformer_obs.csv")

# Geneformer epithelial
for split in ["train_dev", "martin", "oliver_no_martin"]:
    emb, obs = gf_epi_files[split]
    add("Geneformer_frozen", "epithelial", split, emb, obs)

df = pd.DataFrame(registry)

print("\nValidating registry row counts...")
statuses = []
for _, r in df.iterrows():
    emb = Path(r["embedding_file"])
    obs = Path(r["obs_file"])

    status = "OK"
    x_shape = None
    obs_shape = None

    if not emb.exists():
        status = "MISSING_EMB"
    elif not obs.exists():
        status = "MISSING_OBS"
    else:
        X = np.load(emb)
        odf = pd.read_csv(obs)
        x_shape = X.shape
        obs_shape = odf.shape
        if X.shape[0] != len(odf):
            status = "ROW_MISMATCH"

    statuses.append({
        **r.to_dict(),
        "embedding_shape": str(x_shape),
        "obs_shape": str(obs_shape),
        "status": status,
    })

status_df = pd.DataFrame(statuses)
reg_out = OUT / "manual_embedding_registry.csv"
status_out = OUT / "manual_embedding_registry_status.csv"

df.to_csv(reg_out, index=False)
status_df.to_csv(status_out, index=False)

print(status_df.to_string(index=False))
print("\nWrote:")
print(" ", reg_out)
print(" ", status_out)

bad = status_df[status_df["status"] != "OK"]
if len(bad):
    raise SystemExit("\nSome registry entries are not OK. Fix these before running no-leak tests.")
