from pathlib import Path
import numpy as np
import pandas as pd
import anndata as ad
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
)

ROOT = Path.cwd()

IN_DIR = ROOT / "results" / "three_dataset_benchmark"
COMBINED_FILE = IN_DIR / "combined_garrido_kong_smillie_hvg.h5ad"
PCA_FILE = IN_DIR / "X_pca_three_dataset.npy"
SCVI_FILE = IN_DIR / "X_scvi_three_dataset.npy"

OUT_DIR = IN_DIR
OUT_DIR.mkdir(parents=True, exist_ok=True)

MIN_CELLS = 300
MIN_CLASS = 50


def eval_model(X, obs, train_mask, test_mask, model_name, analysis_type, broad_cell_group, train_label, test_label):
    y = obs["y_ibd"].astype(int).values

    if train_mask.sum() < MIN_CELLS or test_mask.sum() < MIN_CELLS:
        return None

    train_counts = pd.Series(y[train_mask]).value_counts().to_dict()
    test_counts = pd.Series(y[test_mask]).value_counts().to_dict()

    for label in [0, 1]:
        if train_counts.get(label, 0) < MIN_CLASS:
            return None
        if test_counts.get(label, 0) < MIN_CLASS:
            return None

    clf = LogisticRegression(
        max_iter=5000,
        class_weight="balanced",
        solver="lbfgs",
    )

    clf.fit(X[train_mask], y[train_mask])

    probs = clf.predict_proba(X[test_mask])[:, 1]
    preds = clf.predict(X[test_mask])

    return {
        "model": model_name,
        "analysis_type": analysis_type,
        "broad_cell_group": broad_cell_group,
        "train_dataset": train_label,
        "test_dataset": test_label,
        "n_train": int(train_mask.sum()),
        "n_test": int(test_mask.sum()),
        "train_normal": int(train_counts.get(0, 0)),
        "train_ibd": int(train_counts.get(1, 0)),
        "test_normal": int(test_counts.get(0, 0)),
        "test_ibd": int(test_counts.get(1, 0)),
        "train_positive_rate": float(y[train_mask].mean()),
        "test_positive_rate": float(y[test_mask].mean()),
        "auroc": roc_auc_score(y[test_mask], probs),
        "auprc": average_precision_score(y[test_mask], probs),
        "balanced_accuracy": balanced_accuracy_score(y[test_mask], preds),
        "f1": f1_score(y[test_mask], preds),
    }


def main():
    if not COMBINED_FILE.exists():
        raise FileNotFoundError(f"Missing {COMBINED_FILE}. Run src\\41_three_dataset_embeddings.py first.")

    adata = ad.read_h5ad(COMBINED_FILE)
    obs = adata.obs.copy()

    models = {}

    if PCA_FILE.exists():
        models["PCA_30"] = np.load(PCA_FILE)

    if SCVI_FILE.exists():
        models["scVI_30"] = np.load(SCVI_FILE)
    else:
        print("WARNING: no scVI embeddings found. Evaluating PCA only.")

    for model_name, X in models.items():
        if X.shape[0] != obs.shape[0]:
            raise ValueError(f"{model_name} rows {X.shape[0]} != obs rows {obs.shape[0]}")

    dataset_values = obs["dataset_eval"].astype(str).values
    group_values = obs["broad_cell_group"].astype(str).values

    datasets = ["garrido", "kong", "smillie"]
    groups = ["epithelial", "myeloid", "lymphoid_plasma", "stromal_endothelial"]

    rows = []

    for model_name, X in models.items():
        print("=" * 100)
        print(f"Evaluating {model_name}")

        # Pairwise full transfer: A -> B.
        for train_dataset in datasets:
            for test_dataset in datasets:
                if train_dataset == test_dataset:
                    continue

                train_mask = dataset_values == train_dataset
                test_mask = dataset_values == test_dataset

                r = eval_model(
                    X, obs, train_mask, test_mask,
                    model_name=model_name,
                    analysis_type="pairwise_full",
                    broad_cell_group="all_cells",
                    train_label=train_dataset,
                    test_label=test_dataset,
                )

                if r is not None:
                    print(f"{model_name} full {train_dataset}→{test_dataset}: AUROC={r['auroc']:.3f}")
                    rows.append(r)

        # Leave-one-dataset-out: train on two, test on one.
        for test_dataset in datasets:
            train_datasets = [d for d in datasets if d != test_dataset]

            train_mask = np.isin(dataset_values, train_datasets)
            test_mask = dataset_values == test_dataset

            train_label = "+".join(train_datasets)

            r = eval_model(
                X, obs, train_mask, test_mask,
                model_name=model_name,
                analysis_type="leave_one_dataset_out",
                broad_cell_group="all_cells",
                train_label=train_label,
                test_label=test_dataset,
            )

            if r is not None:
                print(f"{model_name} LODO {train_label}→{test_dataset}: AUROC={r['auroc']:.3f}")
                rows.append(r)

        # Pairwise cell-type matched.
        for group in groups:
            group_mask = group_values == group

            for train_dataset in datasets:
                for test_dataset in datasets:
                    if train_dataset == test_dataset:
                        continue

                    train_mask = group_mask & (dataset_values == train_dataset)
                    test_mask = group_mask & (dataset_values == test_dataset)

                    r = eval_model(
                        X, obs, train_mask, test_mask,
                        model_name=model_name,
                        analysis_type="pairwise_celltype_matched",
                        broad_cell_group=group,
                        train_label=train_dataset,
                        test_label=test_dataset,
                    )

                    if r is not None:
                        print(f"{model_name} {group} {train_dataset}→{test_dataset}: AUROC={r['auroc']:.3f}")
                        rows.append(r)

        # Leave-one-dataset-out cell-type matched.
        for group in groups:
            group_mask = group_values == group

            for test_dataset in datasets:
                train_datasets = [d for d in datasets if d != test_dataset]

                train_mask = group_mask & np.isin(dataset_values, train_datasets)
                test_mask = group_mask & (dataset_values == test_dataset)

                train_label = "+".join(train_datasets)

                r = eval_model(
                    X, obs, train_mask, test_mask,
                    model_name=model_name,
                    analysis_type="lodo_celltype_matched",
                    broad_cell_group=group,
                    train_label=train_label,
                    test_label=test_dataset,
                )

                if r is not None:
                    print(f"{model_name} LODO {group} {train_label}→{test_dataset}: AUROC={r['auroc']:.3f}")
                    rows.append(r)

    results = pd.DataFrame(rows)
    results = results.sort_values("auroc", ascending=False)

    out_file = OUT_DIR / "three_dataset_benchmark_results.csv"
    results.to_csv(out_file, index=False)

    print("\nSaved:")
    print(out_file)

    print("\nTop results:")
    print(results.head(30))

    # Plot top 20.
    top = results.head(20).iloc[::-1].copy()
    top["label"] = (
        top["model"].astype(str)
        + " | "
        + top["analysis_type"].astype(str)
        + " | "
        + top["broad_cell_group"].astype(str)
        + " | "
        + top["train_dataset"].astype(str)
        + "→"
        + top["test_dataset"].astype(str)
    )

    plt.figure(figsize=(13, 8))
    plt.barh(range(len(top)), top["auroc"])
    plt.axvline(0.5, linestyle="--")
    plt.yticks(range(len(top)), top["label"])
    plt.xlabel("AUROC")
    plt.title("Three-dataset IBD cross-atlas benchmark")
    plt.tight_layout()

    fig_path = OUT_DIR / "three_dataset_top20_auroc.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()

    print("Saved figure:")
    print(fig_path)


if __name__ == "__main__":
    main()
