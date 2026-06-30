from pathlib import Path
import pandas as pd

files = [
    "embeddings_final6/geneformer/train_dev_all_cells_common_geneformer_obs.csv",
    "embeddings_final6/geneformer/locked_test_martin_all_cells_common_geneformer_obs.csv",
    "embeddings_final6/geneformer/locked_test_oliver_no_martin_all_cells_common_geneformer_obs.csv",
]

for f in files:
    p = Path(f)
    print("\n" + "="*100)
    print(p)
    df = pd.read_csv(p)
    print("shape:", df.shape)
    print("columns:", list(df.columns))

    if "y_ibd" in df.columns:
        print("\ny_ibd counts:")
        print(df["y_ibd"].value_counts(dropna=False).to_string())

    for col in ["disease_label_raw", "dataset_id", "dataset_eval", "tissue_label", "broad_cell_group"]:
        if col in df.columns:
            print(f"\n{col} by y_ibd:")
            print(pd.crosstab(df[col], df["y_ibd"]).head(40).to_string())
