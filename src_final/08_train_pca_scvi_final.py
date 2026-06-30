from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import scipy.sparse as sp

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
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
OUT_DIR = ROOT / "results_final6" / "pca_scvi_final"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EMB_DIR = ROOT / "embeddings_final6" / "pca_scvi"
EMB_DIR.mkdir(parents=True, exist_ok=True)

N_HVG = 2500
N_PCA = 50
N_SCVI_LATENT = 30
SCVI_MAX_EPOCHS = 30
SCVI_QUERY_EPOCHS = 20
RANDOM_SEED = 0

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


def make_dense(X):
    if sp.issparse(X):
        return X.toarray()
    return np.asarray(X)


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


def preprocess_for_pca(train, tests):
    """
    PCA pipeline:
    - normalize/log train and tests separately
    - choose HVGs using train only
    - fit scaler/PCA using train only
    - transform locked tests
    """
    train_p = train.copy()
    tests_p = {k: v.copy() for k, v in tests.items()}

    sc.pp.normalize_total(train_p, target_sum=1e4)
    sc.pp.log1p(train_p)

    for t in tests_p.values():
        sc.pp.normalize_total(t, target_sum=1e4)
        sc.pp.log1p(t)

    sc.pp.highly_variable_genes(
        train_p,
        n_top_genes=min(N_HVG, train_p.n_vars),
        flavor="seurat",
        subset=False,
    )

    hvg_mask = train_p.var["highly_variable"].values
    hvg_genes = train_p.var_names[hvg_mask].astype(str).tolist()

    print(f"PCA HVGs selected: {len(hvg_genes)}")

    train_hvg = train_p[:, hvg_genes].copy()
    tests_hvg = {k: v[:, hvg_genes].copy() for k, v in tests_p.items()}

    x_train = make_dense(train_hvg.X)

    scaler = StandardScaler(with_mean=True, with_std=True)
    x_train_scaled = scaler.fit_transform(x_train)

    pca = PCA(n_components=min(N_PCA, x_train_scaled.shape[1]), random_state=RANDOM_SEED)
    z_train = pca.fit_transform(x_train_scaled)

    z_tests = {}
    for name, t in tests_hvg.items():
        x_test = make_dense(t.X)
        x_test_scaled = scaler.transform(x_test)
        z_tests[name] = pca.transform(x_test_scaled)

    return z_train, z_tests, hvg_genes, pca.explained_variance_ratio_


def run_pca(group_name, train, tests):
    print("\n" + "=" * 120)
    print(f"Running PCA for {group_name}")

    z_train, z_tests, hvg_genes, evr = preprocess_for_pca(train, tests)

    y_train = train.obs["y_ibd"].astype(int).values

    rows = []

    np.save(EMB_DIR / f"{group_name}_PCA_train.npy", z_train)
    pd.Series(hvg_genes).to_csv(EMB_DIR / f"{group_name}_PCA_hvg_genes.csv", index=False)
    pd.Series(evr).to_csv(EMB_DIR / f"{group_name}_PCA_explained_variance_ratio.csv", index=False)

    for test_name, z_test in z_tests.items():
        y_test = tests[test_name].obs["y_ibd"].astype(int).values

        np.save(EMB_DIR / f"{group_name}_PCA_test_{test_name}.npy", z_test)

        row = evaluate_classifier(
            model_name="PCA_50",
            group_name=group_name,
            train_name="train_dev",
            test_name=test_name,
            z_train=z_train,
            y_train=y_train,
            z_test=z_test,
            y_test=y_test,
        )
        rows.append(row)

        print(row)

    return rows


def try_import_scvi():
    try:
        import scvi
        return scvi
    except Exception as e:
        print("\nCould not import scvi. Skipping scVI.")
        print("Import error:")
        print(repr(e))
        return None


def preprocess_for_scvi(train, tests):
    """
    scVI pipeline:
    - choose HVGs from normalized/log train only
    - use those genes on raw/count-like matrices
    - train scVI reference on train/dev only
    - map locked tests as query data without labels
    """
    train_for_hvg = train.copy()
    sc.pp.normalize_total(train_for_hvg, target_sum=1e4)
    sc.pp.log1p(train_for_hvg)

    sc.pp.highly_variable_genes(
        train_for_hvg,
        n_top_genes=min(N_HVG, train_for_hvg.n_vars),
        flavor="seurat",
        subset=False,
    )

    hvg_genes = train_for_hvg.var_names[train_for_hvg.var["highly_variable"].values].astype(str).tolist()

    train_scvi = train[:, hvg_genes].copy()
    tests_scvi = {k: v[:, hvg_genes].copy() for k, v in tests.items()}

    # scVI expects nonnegative values. If tiny negatives exist, clip.
    def clean_x(a):
        if sp.issparse(a.X):
            a.X.data[a.X.data < 0] = 0
        else:
            a.X = np.maximum(a.X, 0)
        return a

    train_scvi = clean_x(train_scvi)
    tests_scvi = {k: clean_x(v) for k, v in tests_scvi.items()}

    return train_scvi, tests_scvi, hvg_genes


def run_scvi(group_name, train, tests):
    scvi = try_import_scvi()
    if scvi is None:
        return []

    print("\n" + "=" * 120)
    print(f"Running scVI for {group_name}")

    train_scvi, tests_scvi, hvg_genes = preprocess_for_scvi(train, tests)

    print("scVI train shape:", train_scvi.shape)
    print("scVI test shapes:", {k: v.shape for k, v in tests_scvi.items()})

    scvi.settings.seed = RANDOM_SEED

    scvi.model.SCVI.setup_anndata(train_scvi, batch_key="dataset_eval")

    model = scvi.model.SCVI(
        train_scvi,
        n_latent=N_SCVI_LATENT,
        n_layers=2,
        n_hidden=128,
    )

    model.train(
        max_epochs=SCVI_MAX_EPOCHS,
        early_stopping=True,
        check_val_every_n_epoch=5,
    )

    z_train = model.get_latent_representation(train_scvi)
    y_train = train_scvi.obs["y_ibd"].astype(int).values

    np.save(EMB_DIR / f"{group_name}_scVI_train.npy", z_train)
    pd.Series(hvg_genes).to_csv(EMB_DIR / f"{group_name}_scVI_hvg_genes.csv", index=False)

    model_dir = EMB_DIR / f"{group_name}_scVI_model"
    model.save(model_dir, overwrite=True)

    rows = []

    for test_name, test_scvi in tests_scvi.items():
        print("\n" + "-" * 80)
        print(f"Mapping query test set: {test_name}")

        try:
            scvi.model.SCVI.prepare_query_anndata(test_scvi, model)
            q_model = scvi.model.SCVI.load_query_data(test_scvi, model)

            q_model.train(
                max_epochs=SCVI_QUERY_EPOCHS,
                early_stopping=True,
                check_val_every_n_epoch=5,
            )

            z_test = q_model.get_latent_representation()
        except Exception as e:
            print("Query mapping failed. Trying direct get_latent_representation fallback.")
            print(repr(e))
            z_test = model.get_latent_representation(test_scvi)

        y_test = test_scvi.obs["y_ibd"].astype(int).values

        np.save(EMB_DIR / f"{group_name}_scVI_test_{test_name}.npy", z_test)

        row = evaluate_classifier(
            model_name="scVI_30",
            group_name=group_name,
            train_name="train_dev",
            test_name=test_name,
            z_train=z_train,
            y_train=y_train,
            z_test=z_test,
            y_test=y_test,
        )
        rows.append(row)

        print(row)

    return rows


def main():
    all_rows = []

    for group_name, cfg in TASKS.items():
        print("\n" + "#" * 120)
        print(f"TASK GROUP: {group_name}")
        print("#" * 120)

        train = ad.read_h5ad(cfg["train"])
        tests = {k: ad.read_h5ad(v) for k, v in cfg["tests"].items()}

        print("Train:", train.shape)
        print(train.obs["dataset_eval"].value_counts())
        print(train.obs["y_ibd"].value_counts())

        for name, t in tests.items():
            print(f"Test {name}:", t.shape)
            print(t.obs["y_ibd"].value_counts())

        all_rows.extend(run_pca(group_name, train, tests))
        all_rows.extend(run_scvi(group_name, train, tests))

    results = pd.DataFrame(all_rows)
    results = results.sort_values(["group", "test_dataset", "model"])

    out_csv = OUT_DIR / "pca_scvi_locked_test_results.csv"
    results.to_csv(out_csv, index=False)

    print("\n" + "=" * 120)
    print("FINAL PCA/scVI LOCKED TEST RESULTS:")
    print(results.to_string(index=False))
    print("\nSaved:")
    print(out_csv)


if __name__ == "__main__":
    main()