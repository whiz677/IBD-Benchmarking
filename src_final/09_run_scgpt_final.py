from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import os
import json
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    accuracy_score,
)


ROOT = Path.cwd()

DATA_DIR = ROOT / "data" / "final_benchmark_matrices"
OUT_DIR = ROOT / "results_final6" / "scgpt_final"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EMB_DIR = ROOT / "embeddings_final6" / "scgpt"
EMB_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 0
BATCH_SIZE = 8
MAX_LENGTH = 1200

TASKS = {
    "all_cells": {
        "train": DATA_DIR / "train_dev_all_cells_common.h5ad",
        "tests": {
            "martin": DATA_DIR / "locked_test_martin_all_cells_common.h5ad",
            "oliver_no_martin": DATA_DIR / "locked_test_oliver_no_martin_all_cells_common.h5ad",
        },
    },
    "epithelial": {
        "train": DATA_DIR / "train_dev_epithelial_common.h5ad",
        "tests": {
            "martin": DATA_DIR / "locked_test_martin_epithelial_common.h5ad",
            "oliver_no_martin": DATA_DIR / "locked_test_oliver_no_martin_epithelial_common.h5ad",
        },
    },
}


class SimpleGeneVocab:
    """
    Pure-Python replacement for scGPT GeneVocab when torchtext.vocab.vocab is broken.
    It supports the methods scGPT embed_data actually uses:
    - from_file
    - __contains__
    - __getitem__
    - __call__
    - append_token
    - set_default_index
    - get_stoi
    - len()
    """

    def __init__(self, token2idx=None):
        self.default_index = None

        if token2idx is None:
            token2idx = {}

        if isinstance(token2idx, list):
            self.token2idx = {str(tok): i for i, tok in enumerate(token2idx)}
        elif isinstance(token2idx, dict):
            self.token2idx = {str(k): int(v) for k, v in token2idx.items()}
        else:
            raise TypeError("SimpleGeneVocab needs list or dict.")

        self._rebuild_itos()

    def _rebuild_itos(self):
        if len(self.token2idx) == 0:
            self.idx2token = []
            return

        max_idx = max(self.token2idx.values())
        self.idx2token = ["<unk>"] * (max_idx + 1)

        for tok, idx in self.token2idx.items():
            if idx >= 0:
                if idx >= len(self.idx2token):
                    self.idx2token.extend(["<unk>"] * (idx + 1 - len(self.idx2token)))
                self.idx2token[idx] = tok

    @classmethod
    def from_file(cls, file_path):
        file_path = Path(file_path)

        if file_path.suffix.lower() != ".json":
            raise ValueError(f"Only .json vocab supported by SimpleGeneVocab, got {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            token2idx = json.load(f)

        return cls(token2idx)

    @classmethod
    def from_dict(cls, token2idx, default_token="<pad>"):
        vocab = cls(token2idx)
        if default_token is not None and default_token in vocab:
            vocab.set_default_index(vocab[default_token])
        return vocab

    def __len__(self):
        return len(self.token2idx)

    def __contains__(self, token):
        return str(token) in self.token2idx

    def __getitem__(self, token):
        token = str(token)
        if token in self.token2idx:
            return self.token2idx[token]
        if self.default_index is not None:
            return self.default_index
        raise KeyError(token)

    def __call__(self, tokens):
        return [self[t] for t in tokens]

    def append_token(self, token):
        token = str(token)
        if token not in self.token2idx:
            self.token2idx[token] = len(self.token2idx)
            self._rebuild_itos()

    def insert_token(self, token, index):
        token = str(token)
        index = int(index)

        if token in self.token2idx:
            return

        # If index is already occupied, shift existing tokens upward.
        occupied = {v: k for k, v in self.token2idx.items()}
        if index in occupied:
            for tok, idx in list(self.token2idx.items()):
                if idx >= index:
                    self.token2idx[tok] = idx + 1

        self.token2idx[token] = index
        self._rebuild_itos()

    def set_default_index(self, index):
        self.default_index = int(index)

    def set_default_token(self, token):
        self.set_default_index(self[token])

    def get_stoi(self):
        return dict(self.token2idx)

    def get_itos(self):
        return list(self.idx2token)


def patch_scgpt_runtime():
    """
    Fixes two Windows/scGPT problems:
    1. torchtext.vocab.vocab missing.
    2. os.sched_getaffinity missing / Windows DataLoader pickling issues.

    We patch scGPT's cell_emb.GeneVocab to SimpleGeneVocab so embed_data
    never touches the broken torchtext.vocab.vocab function.
    """

    # Force scGPT DataLoader num_workers=0 on Windows.
    # scGPT uses len(os.sched_getaffinity(0)); returning empty set makes num_workers=0.
    os.sched_getaffinity = lambda pid: set()

    import scgpt as scg
    import scgpt.tasks.cell_emb as cell_emb

    cell_emb.GeneVocab = SimpleGeneVocab

    try:
        import scgpt.tokenizer.gene_tokenizer as gene_tokenizer
        gene_tokenizer.GeneVocab = SimpleGeneVocab
    except Exception:
        pass

    try:
        import scgpt.tokenizer as tokenizer_pkg
        tokenizer_pkg.GeneVocab = SimpleGeneVocab
    except Exception:
        pass

    print("Patched scGPT runtime:")
    print("- using SimpleGeneVocab instead of broken torchtext GeneVocab")
    print("- forcing DataLoader num_workers=0 for Windows")

    return scg


def find_scgpt_model_dir():
    candidates = []

    base_dirs = [
        ROOT / "models" / "scGPT_human",
        ROOT / "models",
        ROOT / "models" / "scGPT",
        ROOT / "models" / "scgpt",
    ]

    for base in base_dirs:
        if base.exists():
            candidates.append(base)
            candidates.extend([p for p in base.rglob("*") if p.is_dir()])

    valid = []

    for d in candidates:
        try:
            files = {p.name for p in d.iterdir() if p.is_file()}
        except Exception:
            continue

        has_args = "args.json" in files
        has_vocab = "vocab.json" in files
        has_best_model = "best_model.pt" in files

        if has_args and has_vocab and has_best_model:
            valid.append(d)

    # Remove duplicates while preserving order.
    seen = set()
    clean_valid = []
    for v in valid:
        if str(v) not in seen:
            clean_valid.append(v)
            seen.add(str(v))

    if len(clean_valid) == 0:
        print("Could not find scGPT model directory.")
        print("Need folder containing args.json, vocab.json, and best_model.pt.")
        print("Searched under:")
        print(ROOT / "models")
        raise FileNotFoundError("No valid scGPT model directory found.")

    print("Found possible scGPT model directories:")
    for v in clean_valid:
        print(v)

    return clean_valid[0]


def prepare_for_scgpt(adata):
    adata = adata.copy()

    adata.var_names = adata.var_names.astype(str)
    adata.var_names_make_unique()

    # final matrices already use harmonized gene symbols as var_names
    adata.var["gene_name"] = adata.var_names.astype(str)

    # scGPT expects nonnegative expression
    if sp.issparse(adata.X):
        adata.X.data[adata.X.data < 0] = 0
    else:
        adata.X = np.maximum(np.asarray(adata.X), 0)

    return adata


def extract_embedding_matrix(embed_adata):
    # If return_new_adata=True, scGPT returns embeddings as .X
    if embed_adata.X is not None:
        x = embed_adata.X
        if sp.issparse(x):
            x = x.toarray()
        x = np.asarray(x)

        if x.ndim == 2 and x.shape[0] == embed_adata.n_obs and x.shape[1] > 1:
            print(f"Using embedding matrix from returned AnnData.X: {x.shape}")
            return x.astype(np.float32)

    # Some versions store in obsm instead
    possible_keys = [
        "X_scGPT",
        "X_scgpt",
        "scGPT",
        "scgpt",
        "X_cell_emb",
        "X_emb",
    ]

    for key in possible_keys:
        if key in embed_adata.obsm:
            z = np.asarray(embed_adata.obsm[key])
            print(f"Using embedding key obsm['{key}']: {z.shape}")
            return z.astype(np.float32)

    print("Available obsm keys:")
    print(list(embed_adata.obsm.keys()))
    raise ValueError("Could not find scGPT embedding matrix.")


def embed_with_scgpt(adata, model_dir, label):
    import torch

    scg = patch_scgpt_runtime()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("\n" + "=" * 120)
    print(f"Embedding {label} with scGPT")
    print(f"Input shape: {adata.shape}")
    print(f"Device: {device}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Max length: {MAX_LENGTH}")

    adata = prepare_for_scgpt(adata)

    # Quick vocabulary overlap check before running slow embedding.
    vocab_path = Path(model_dir) / "vocab.json"
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab_json = json.load(f)

    genes = set(adata.var["gene_name"].astype(str))
    vocab_genes = set(str(x) for x in vocab_json.keys())
    overlap = genes.intersection(vocab_genes)
    print(f"Gene overlap with scGPT vocab: {len(overlap):,}/{adata.n_vars:,}")

    if len(overlap) < 500:
        raise ValueError(
            f"Too few genes overlap scGPT vocab: {len(overlap)}. "
            "Check gene symbols in final matrices."
        )

    embed_adata = scg.tasks.embed_data(
        adata,
        model_dir,
        gene_col="gene_name",
        max_length=MAX_LENGTH,
        batch_size=BATCH_SIZE,
        obs_to_save=[
            "dataset_eval",
            "dataset_id",
            "role",
            "y_ibd",
            "disease_label_raw",
            "cell_type_label",
            "broad_cell_group",
            "donor_label",
            "tissue_label",
        ],
        device=device,
        use_fast_transformer=False,
        return_new_adata=True,
    )

    z = extract_embedding_matrix(embed_adata)

    if z.shape[0] != adata.n_obs:
        raise ValueError(f"Embedding row mismatch for {label}: {z.shape[0]} vs {adata.n_obs}")

    return z


def evaluate_classifier(model_name, group_name, train_name, test_name, z_train, y_train, z_test, y_test):
    clf = LogisticRegression(
        max_iter=5000,
        class_weight="balanced",
        solver="lbfgs",
        random_state=RANDOM_SEED,
    )

    clf.fit(z_train, y_train)

    probs = clf.predict_proba(z_test)[:, 1]
    preds = clf.predict(z_test)

    return {
        "model": model_name,
        "group": group_name,
        "train_dataset": train_name,
        "test_dataset": test_name,
        "n_train": len(y_train),
        "n_test": len(y_test),
        "train_positive_rate": float(np.mean(y_train)),
        "test_positive_rate": float(np.mean(y_test)),
        "auroc": float(roc_auc_score(y_test, probs)),
        "auprc": float(average_precision_score(y_test, probs)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, preds)),
        "accuracy": float(accuracy_score(y_test, preds)),
        "f1": float(f1_score(y_test, preds)),
    }


def main():
    model_dir = find_scgpt_model_dir()

    print("\nUsing scGPT model directory:")
    print(model_dir)

    all_rows = []

    for group_name, cfg in TASKS.items():
        print("\n" + "#" * 120)
        print(f"FINAL scGPT TASK GROUP: {group_name}")
        print("#" * 120)

        train = ad.read_h5ad(cfg["train"])
        tests = {k: ad.read_h5ad(v) for k, v in cfg["tests"].items()}

        print("Train:", train.shape)
        print(train.obs["dataset_eval"].value_counts())
        print(train.obs["y_ibd"].value_counts())

        for test_name, test in tests.items():
            print(f"\nTest {test_name}: {test.shape}")
            print(test.obs["y_ibd"].value_counts())

        z_train_path = EMB_DIR / f"{group_name}_scGPT_train.npy"

        if z_train_path.exists():
            print(f"\nLoading cached train embeddings: {z_train_path}")
            z_train = np.load(z_train_path)
        else:
            z_train = embed_with_scgpt(train, model_dir, f"{group_name}_train")
            np.save(z_train_path, z_train)

        y_train = train.obs["y_ibd"].astype(int).values

        for test_name, test in tests.items():
            z_test_path = EMB_DIR / f"{group_name}_scGPT_test_{test_name}.npy"

            if z_test_path.exists():
                print(f"\nLoading cached test embeddings: {z_test_path}")
                z_test = np.load(z_test_path)
            else:
                z_test = embed_with_scgpt(test, model_dir, f"{group_name}_test_{test_name}")
                np.save(z_test_path, z_test)

            y_test = test.obs["y_ibd"].astype(int).values

            row = evaluate_classifier(
                model_name="scGPT_frozen",
                group_name=group_name,
                train_name="train_dev",
                test_name=test_name,
                z_train=z_train,
                y_train=y_train,
                z_test=z_test,
                y_test=y_test,
            )

            all_rows.append(row)
            print("\nResult:")
            print(row)

    results = pd.DataFrame(all_rows)
    results = results.sort_values(["group", "test_dataset", "model"])

    out_csv = OUT_DIR / "scgpt_locked_test_results.csv"
    results.to_csv(out_csv, index=False)

    print("\n" + "=" * 120)
    print("FINAL scGPT LOCKED TEST RESULTS:")
    print(results.to_string(index=False))
    print("\nSaved:")
    print(out_csv)


if __name__ == "__main__":
    main()