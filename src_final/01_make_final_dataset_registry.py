from pathlib import Path
import pandas as pd

ROOT = Path.cwd()

OUT_DIR = ROOT / "results_final6"
OUT_DIR.mkdir(parents=True, exist_ok=True)

REGISTRY_FILE = OUT_DIR / "dataset_registry.csv"

rows = [
    {
        "dataset_id": "garrido",
        "dataset_name": "Garrido-Trigo IBD healthy colonic mucosa",
        "role": "train_dev",
        "status": "ready_raw_or_processed",
        "expected_path": "data/raw/cellxgene/ibd_garrido_trigo",
        "description": "Original IBD/healthy colonic mucosa dataset. Mixed cell types. Used as core train/dev cohort.",
    },
    {
        "dataset_id": "kong",
        "dataset_name": "Kong Crohn normal ileum colon atlas",
        "role": "train_dev",
        "status": "ready_processed",
        "expected_path": "data/processed/ibd_kong_2023_processed.h5ad",
        "description": "Crohn disease/normal ileum-colon atlas. Mixed epithelial, immune, stromal compartments.",
    },
    {
        "dataset_id": "smillie_epi",
        "dataset_name": "Smillie SCP259 UC healthy epithelial colon",
        "role": "train_dev",
        "status": "ready_processed",
        "expected_path": "data/processed/ibd_smillie_uc_processed.h5ad",
        "description": "UC/healthy colon epithelial subset from SCP259. Epithelial-only but balanced disease/control.",
    },
    {
        "dataset_id": "mitsialis",
        "dataset_name": "Mitsialis UC/CD colon blood immune dataset",
        "role": "train_dev",
        "status": "needs_processing",
        "expected_path": "data/raw/ibd_mitsialis_2020",
        "description": "Candidate fourth train/dev cohort. Needs metadata inspection before deciding exact binary labels.",
    },
    {
        "dataset_id": "martin",
        "dataset_name": "Martin Crohn lesion anti-TNF dataset",
        "role": "locked_test",
        "status": "needs_processing",
        "expected_path": "data/raw/ibd_martin_2019",
        "description": "Locked external test. Likely inflamed vs uninvolved Crohn lesion or disease-module validation.",
    },
    {
        "dataset_id": "oliver",
        "dataset_name": "Oliver 2024 inflammatory gut disease atlas",
        "role": "locked_test",
        "status": "needs_processing",
        "expected_path": "data/raw/ibd_oliver_2024_cellxgene/manual_download",
        "description": "Locked external stress test. Use large intestine and small intestine h5ad files; inspect overlap before final use.",
    },
]

df = pd.DataFrame(rows)

def path_exists(p):
    path = ROOT / p
    return path.exists()

df["path_exists"] = df["expected_path"].map(path_exists)

df.to_csv(REGISTRY_FILE, index=False)

print("Saved dataset registry:")
print(REGISTRY_FILE)

print("\nRegistry:")
print(df.to_string(index=False))

missing = df[~df["path_exists"]]
if len(missing) > 0:
    print("\nWARNING: Some expected paths are missing:")
    print(missing[["dataset_id", "expected_path"]].to_string(index=False))
else:
    print("\nAll expected paths exist.")