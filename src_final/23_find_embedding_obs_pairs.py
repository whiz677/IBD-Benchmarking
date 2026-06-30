from pathlib import Path
import pandas as pd

roots = [
    Path("embeddings"),
    Path("embeddings_final6"),
    Path("results"),
    Path("results_final6"),
]

rows = []

def infer_model(path):
    s = str(path).lower()
    if "geneformer" in s:
        return "Geneformer_frozen"
    if "scgpt" in s:
        return "scGPT_frozen"
    if "scvi" in s:
        return "scVI"
    if "pca" in s:
        return "PCA"
    return "unknown"

def infer_group(path):
    s = str(path).lower()
    if "epithelial" in s or "_epi" in s:
        return "epithelial"
    return "all_cells"

def infer_split(path):
    s = str(path).lower()
    if "train_dev" in s or "train" in s:
        return "train_dev"
    if "martin" in s and "oliver" not in s:
        return "martin"
    if "oliver" in s:
        return "oliver_no_martin"
    return "unknown"

for root in roots:
    if not root.exists():
        continue

    for emb in root.rglob("*.npy"):
        if "site-packages" in str(emb):
            continue

        name = emb.name
        possible_obs = []

        if name.endswith("_embeddings.npy"):
            possible_obs.append(emb.with_name(name.replace("_embeddings.npy", "_obs.csv")))

        if name.endswith("_geneformer_embeddings.npy"):
            possible_obs.append(emb.with_name(name.replace("_geneformer_embeddings.npy", "_geneformer_obs.csv")))

        possible_obs.append(emb.with_name(emb.stem + "_obs.csv"))
        possible_obs.append(emb.parent / "obs.csv")

        found_obs = None
        for o in possible_obs:
            if o.exists():
                found_obs = o
                break

        rows.append({
            "model": infer_model(emb),
            "group": infer_group(emb),
            "split": infer_split(emb),
            "embedding_file": str(emb),
            "obs_file": str(found_obs) if found_obs else "",
            "has_obs": found_obs is not None,
        })

df = pd.DataFrame(rows)
out = Path("results_final6/proper_no_leak_benchmark/embedding_registry_detected.csv")
out.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out, index=False)

print(df.to_string(index=False))
print("\nWrote:", out)
