from pathlib import Path
import scanpy as sc
import anndata as ad
from geneformer import TranscriptomeTokenizer

OUT = Path("finetune_models/outputs/geneformer_locked_external")
H5AD_IN = OUT / "h5ad_input"
TRAIN_DIR = H5AD_IN / "train_only"
TEST_DIR = H5AD_IN / "test_only"
TOK_OUT = OUT / "tokenized"

TRAIN_DIR.mkdir(parents=True, exist_ok=True)
TEST_DIR.mkdir(parents=True, exist_ok=True)
TOK_OUT.mkdir(parents=True, exist_ok=True)

train_file = "data/geneformer_final_inputs/train_dev_all_cells_common_geneformer_ready.h5ad"
test_files = [
    "data/geneformer_final_inputs/locked_test_martin_all_cells_common_geneformer_ready.h5ad",
    "data/geneformer_final_inputs/locked_test_oliver_no_martin_all_cells_common_geneformer_ready.h5ad",
]

def clean_obs(a, split, test_file):
    a.obs["split"] = split
    a.obs["test_file"] = test_file
    for col in ["y_ibd", "dataset_eval", "cell_type_label", "test_file", "split"]:
        if col in a.obs:
            a.obs[col] = a.obs[col].astype(str)
    return a

print("Loading train...")
train = sc.read_h5ad(train_file)
train = clean_obs(train, "train", "train")

if "ensembl_id" not in train.var.columns:
    raise ValueError("Train file missing var['ensembl_id']")

print("Loading tests...")
tests = []
test_var_template = None

for p in test_files:
    x = sc.read_h5ad(p)
    x = clean_obs(x, "test", Path(p).stem)

    if "ensembl_id" not in x.var.columns:
        raise ValueError(f"Test file missing var['ensembl_id']: {p}")

    if test_var_template is None:
        test_var_template = x.var.copy()

    tests.append(x)

test = ad.concat(tests, join="inner", index_unique="-test")

# anndata.concat can drop var metadata; restore it for the shared genes.
test.var = test_var_template.loc[test.var_names].copy()

print("train", train)
print(train.obs["dataset_eval"].value_counts())
print(train.obs["y_ibd"].value_counts())
print("train var columns:", train.var.columns.tolist())

print("test", test)
print(test.obs["dataset_eval"].value_counts())
print(test.obs["y_ibd"].value_counts())
print("test var columns:", test.var.columns.tolist())

if "ensembl_id" not in test.var.columns:
    raise ValueError("Concatenated test missing var['ensembl_id'] after restore")

train_path = TRAIN_DIR / "train_geneformer_locked.h5ad"
test_path = TEST_DIR / "test_geneformer_locked.h5ad"

train.write_h5ad(train_path)
test.write_h5ad(test_path)

attrs = {
    "y_ibd": "y_ibd",
    "dataset_eval": "dataset_eval",
    "cell_type_label": "cell_type_label",
    "test_file": "test_file",
    "split": "split",
}

tk = TranscriptomeTokenizer(attrs, nproc=8)

print("Tokenizing train only...")
tk.tokenize_data(str(TRAIN_DIR), str(TOK_OUT), "train_locked", file_format="h5ad")

print("Tokenizing test only...")
tk.tokenize_data(str(TEST_DIR), str(TOK_OUT), "test_locked", file_format="h5ad")

print("Done tokenizing:", TOK_OUT)