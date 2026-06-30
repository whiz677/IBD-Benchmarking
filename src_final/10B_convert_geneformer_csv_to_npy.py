from pathlib import Path
import numpy as np
import pandas as pd

OUT_DIR = Path("embeddings_final6/geneformer")

PREFIXES = [
    "train_dev_all_cells_common_geneformer",
    "locked_test_martin_all_cells_common_geneformer",
    "locked_test_oliver_no_martin_all_cells_common_geneformer",
]

for prefix in PREFIXES:
    emb_csv = OUT_DIR / f"{prefix}.csv"
    obs_csv = OUT_DIR / f"{prefix}_obs.csv"
    out_npy = OUT_DIR / f"{prefix}_embeddings.npy"

    print("\n" + "=" * 90)
    print("Processing:", prefix)

    if not emb_csv.exists():
        print("MISSING embedding CSV:", emb_csv)
        continue

    if not obs_csv.exists():
        print("MISSING obs CSV:", obs_csv)
        continue

    print("Reading embedding CSV:", emb_csv)
    df = pd.read_csv(emb_csv)

    print("CSV shape:", df.shape)
    print("Columns preview:", list(df.columns[:10]))

    # Keep only numeric columns. Geneformer embeddings should be numeric.
    num = df.select_dtypes(include=[np.number]).copy()

    # Drop obvious index columns if they exist.
    for c in ["Unnamed: 0", "index", "cell_index"]:
        if c in num.columns:
            num = num.drop(columns=[c])

    print("Numeric embedding matrix shape:", num.shape)

    if num.shape[1] < 16:
        print("ERROR: Too few numeric columns. Here is the CSV head:")
        print(df.head())
        raise SystemExit(1)

    obs = pd.read_csv(obs_csv)
    print("Obs shape:", obs.shape)

    if num.shape[0] != obs.shape[0]:
        print("WARNING: embedding rows != obs rows")
        print("emb rows:", num.shape[0])
        print("obs rows:", obs.shape[0])

    arr = num.to_numpy(dtype=np.float32)
    np.save(out_npy, arr)

    print("WROTE:", out_npy)
    print("NPY shape:", arr.shape)

print("\nDONE converting Geneformer CSVs to NPY.")
