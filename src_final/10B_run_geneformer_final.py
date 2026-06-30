from pathlib import Path
import os
import shutil
import sys
import json

import anndata as ad
import numpy as np
import pandas as pd

ROOT = Path.cwd()

INPUT_DIR = ROOT / "data" / "geneformer_final_inputs"
TOKEN_DIR = ROOT / "data" / "geneformer_tokenized_final"
TMP_DIR = ROOT / "data" / "_geneformer_tmp_10B"
OUT_DIR = ROOT / "embeddings_final6" / "geneformer"

TOKEN_DIR.mkdir(parents=True, exist_ok=True)
TMP_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_INPUTS = [
    "train_dev_all_cells_common_geneformer_ready.h5ad",
    "locked_test_martin_all_cells_common_geneformer_ready.h5ad",
    "locked_test_oliver_no_martin_all_cells_common_geneformer_ready.h5ad",
]

def find_geneformer_model_dir():
    """
    Finds the actual pretrained Geneformer checkpoint folder.
    It must contain config.json and usually pytorch_model.bin/model.safetensors.
    """
    candidates = []

    for cfg in ROOT.rglob("config.json"):
        folder = cfg.parent
        low = str(folder).lower()

        has_weights = (
            (folder / "pytorch_model.bin").exists()
            or (folder / "model.safetensors").exists()
            or any(folder.glob("*.bin"))
            or any(folder.glob("*.safetensors"))
        )

        if has_weights and ("geneformer" in low or "model" in low):
            candidates.append(folder)

    if candidates:
        candidates = sorted(candidates, key=lambda p: len(str(p)))
        print("Using Geneformer checkpoint:", candidates[0])
        return candidates[0]

    print("\nERROR: Could not find a pretrained Geneformer checkpoint folder.")
    print("I found no folder with config.json + model weights.")
    print("\nRun this to inspect your models folder:")
    print("find models -maxdepth 5 -type f | sort | head -200")
    print("\nYou need a folder containing config.json and pytorch_model.bin or model.safetensors.")
    sys.exit(1)

def check_input_h5ad(path):
    print("\nChecking:", path)
    a = ad.read_h5ad(path, backed="r")
    print("  shape:", a.shape)
    print("  obs cols:", list(a.obs.columns)[:25])
    print("  var cols:", list(a.var.columns)[:25])
    print("  first var_names:", list(a.var_names[:5]))

    ok = False
    if "ensembl_id" in a.var.columns:
        vals = list(a.var["ensembl_id"].astype(str).head(5))
        print("  first ensembl_id:", vals)
        ok = any(v.startswith("ENSG") or v.startswith("ENSMUSG") for v in vals)

    if not ok:
        vals = list(map(str, a.var_names[:5]))
        ok = any(v.startswith("ENSG") or v.startswith("ENSMUSG") for v in vals)

    a.file.close()

    if not ok:
        print("\nERROR: This input does not look Geneformer-ready.")
        print("Expected Ensembl IDs in var['ensembl_id'] or var_names.")
        print("Bad file:", path)
        sys.exit(1)

def safe_link_or_copy(src, dst):
    if dst.exists():
        dst.unlink()
    try:
        os.symlink(src, dst)
    except Exception:
        shutil.copy2(src, dst)

def get_attr_dict(h5ad_path):
    """
    Carries useful metadata into the tokenized dataset.
    """
    a = ad.read_h5ad(h5ad_path, backed="r")
    cols = list(a.obs.columns)
    a.file.close()

    wanted = [
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
        "label",
        "disease",
        "condition",
        "dataset",
        "donor",
        "sample",
        "sample_id",
        "patient",
        "patient_id",
        "cell_type",
        "celltype",
        "batch",
        "study",
    ]

    return {c: c for c in wanted if c in cols}

def save_obs(h5ad_path, out_obs):
    a = ad.read_h5ad(h5ad_path, backed="r")
    obs = a.obs.copy()
    a.file.close()
    obs.to_csv(out_obs)
    print("Wrote obs:", out_obs, obs.shape)

def find_embedding_csv(prefix):
    csvs = sorted(OUT_DIR.glob(f"{prefix}*.csv"))
    if csvs:
        return csvs[-1]
    return None

def convert_embedding_csv_to_npy(prefix, h5ad_path):
    """
    Geneformer EmbExtractor often writes a CSV.
    This converts numeric embedding columns to .npy and saves obs separately.
    """
    final_npy = OUT_DIR / f"{prefix}_embeddings.npy"
    final_obs = OUT_DIR / f"{prefix}_obs.csv"

    if not final_obs.exists():
        save_obs(h5ad_path, final_obs)

    if final_npy.exists():
        print("Embeddings already converted:", final_npy)
        return

    csv_path = find_embedding_csv(prefix)

    if csv_path is None:
        print("\nWARNING: No CSV embedding file found to convert for:", prefix)
        print("This may still be okay if Geneformer wrote another format.")
        print("Current output files:")
        for p in sorted(OUT_DIR.glob("*")):
            print(" ", p.name)
        return

    print("Converting Geneformer CSV to NPY:", csv_path)
    df = pd.read_csv(csv_path)

    numeric = df.select_dtypes(include=[np.number])

    # Drop obvious metadata numeric columns if present.
    maybe_meta = [
        "Unnamed: 0",
        "index",
        "cell_index",
        "y_ibd",
        "n_counts",
    ]
    drop_cols = [c for c in maybe_meta if c in numeric.columns]
    if drop_cols:
        numeric = numeric.drop(columns=drop_cols)

    if numeric.shape[1] < 32:
        print("WARNING: numeric embedding dimension looks small:", numeric.shape)
        print("Keeping CSV, but not saving NPY because this may be metadata.")
        return

    arr = numeric.to_numpy(dtype=np.float32)
    np.save(final_npy, arr)
    print("Wrote embeddings:", final_npy, arr.shape)

def run_one(h5ad_path, model_dir):
    from geneformer import TranscriptomeTokenizer, EmbExtractor

    h5ad_path = Path(h5ad_path)
    base = h5ad_path.stem.replace("_geneformer_ready", "")
    token_prefix = base
    emb_prefix = f"{base}_geneformer"

    final_obs = OUT_DIR / f"{emb_prefix}_obs.csv"
    final_npy = OUT_DIR / f"{emb_prefix}_embeddings.npy"

    if final_obs.exists() and final_npy.exists():
        print("\nSkipping existing final output:", base)
        print(" ", final_obs)
        print(" ", final_npy)
        return

    print("\n" + "=" * 100)
    print("10B RUNNING:", base)
    print("=" * 100)

    check_input_h5ad(h5ad_path)

    one_dir = TMP_DIR / base
    if one_dir.exists():
        shutil.rmtree(one_dir)
    one_dir.mkdir(parents=True, exist_ok=True)

    safe_link_or_copy(h5ad_path, one_dir / h5ad_path.name)

    attr_dict = get_attr_dict(h5ad_path)
    print("Metadata attrs carried:", attr_dict)

    dataset_path = TOKEN_DIR / f"{token_prefix}.dataset"

    if dataset_path.exists():
        print("Tokenized dataset already exists:", dataset_path)
    else:
        print("\nTokenizing:", h5ad_path.name)

        tk = TranscriptomeTokenizer(
            custom_attr_name_dict=attr_dict,
            nproc=max(1, min(16, os.cpu_count() or 1)),
        )

        tk.tokenize_data(
            str(one_dir),
            str(TOKEN_DIR),
            token_prefix,
            file_format="h5ad",
        )

    if not dataset_path.exists():
        print("\nERROR: Expected tokenized dataset was not created:")
        print(dataset_path)
        print("\nToken directory contents:")
        for p in sorted(TOKEN_DIR.glob("*")):
            print(" ", p)
        sys.exit(1)

    print("Tokenized dataset:", dataset_path)

    print("\nExtracting embeddings with Geneformer...")

    embex = EmbExtractor(
        model_type="Pretrained",
        num_classes=0,
        emb_mode="cell",
        emb_layer=-1,
        max_ncells=None,
        forward_batch_size=16,
        nproc=max(1, min(16, os.cpu_count() or 1)),
    )

    embex.extract_embs(
        str(model_dir),
        str(dataset_path),
        str(OUT_DIR),
        emb_prefix,
    )

    convert_embedding_csv_to_npy(emb_prefix, h5ad_path)

def main():
    print("ROOT:", ROOT)
    print("INPUT_DIR:", INPUT_DIR)
    print("TOKEN_DIR:", TOKEN_DIR)
    print("OUT_DIR:", OUT_DIR)

    model_dir = find_geneformer_model_dir()

    input_files = []
    for fname in TARGET_INPUTS:
        p = INPUT_DIR / fname
        if not p.exists():
            print("\nERROR: Missing required input:")
            print(p)
            print("\nAvailable files:")
            for x in sorted(INPUT_DIR.glob("*")):
                print(" ", x.name)
            sys.exit(1)
        input_files.append(p)

    print("\n10B will process these all-cells files:")
    for p in input_files:
        print(" ", p)

    for p in input_files:
        run_one(p, model_dir)

    print("\nDONE 10B.")
    print("Final output folder:", OUT_DIR)
    for p in sorted(OUT_DIR.glob("*")):
        print(" ", p.name)

if __name__ == "__main__":
    main()
