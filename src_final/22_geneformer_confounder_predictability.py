from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split

EMB_DIR = Path("embeddings_final6/geneformer")
OUT = Path("results_final6/geneformer_final/confounder_predictability")
OUT.mkdir(parents=True, exist_ok=True)

datasets = {
    "train_dev": (
        EMB_DIR / "train_dev_all_cells_common_geneformer_embeddings.npy",
        EMB_DIR / "train_dev_all_cells_common_geneformer_obs.csv",
    ),
    "martin": (
        EMB_DIR / "locked_test_martin_all_cells_common_geneformer_embeddings.npy",
        EMB_DIR / "locked_test_martin_all_cells_common_geneformer_obs.csv",
    ),
    "oliver_no_martin": (
        EMB_DIR / "locked_test_oliver_no_martin_all_cells_common_geneformer_embeddings.npy",
        EMB_DIR / "locked_test_oliver_no_martin_all_cells_common_geneformer_obs.csv",
    ),
}

target_cols = [
    "y_ibd",
    "dataset_id",
    "dataset_eval",
    "broad_cell_group",
    "cell_type_label",
    "tissue_label",
    "donor_label",
    "sample_label",
    "source_id",
]

rows = []

for dataset_name, (emb_path, obs_path) in datasets.items():
    print("\n" + "="*100)
    print(dataset_name)

    X = np.load(emb_path)
    obs = pd.read_csv(obs_path)

    for c in ["Unnamed: 0", "index"]:
        if c in obs.columns:
            obs = obs.drop(columns=[c])

    if X.shape[0] != len(obs):
        raise ValueError(f"Row mismatch {dataset_name}: {X.shape}, {obs.shape}")

    max_n = 30000
    if X.shape[0] > max_n:
        rng = np.random.default_rng(42)
        idx = rng.choice(X.shape[0], size=max_n, replace=False)
        X_use = X[idx]
        obs_use = obs.iloc[idx].reset_index(drop=True)
    else:
        X_use = X
        obs_use = obs

    for target in target_cols:
        if target not in obs_use.columns:
            continue

        y_raw = obs_use[target].astype(str).fillna("NA")
        counts = y_raw.value_counts()

        valid_classes = counts[counts >= 20].index
        mask = y_raw.isin(valid_classes).values

        if len(valid_classes) < 2 or mask.sum() < 100:
            continue

        X_t = X_use[mask]
        y_raw_t = y_raw[mask]

        le = LabelEncoder()
        y = le.fit_transform(y_raw_t)

        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X_t, y, test_size=0.25, random_state=42, stratify=y
            )
        except Exception:
            X_train, X_test, y_train, y_test = train_test_split(
                X_t, y, test_size=0.25, random_state=42
            )

        clf = Pipeline([
            ("scaler", StandardScaler()),
            ("logreg", LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                solver="lbfgs",
                n_jobs=-1,
                random_state=42,
            )),
        ])

        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)

        rows.append({
            "dataset": dataset_name,
            "target": target,
            "n_cells_used": int(mask.sum()),
            "n_classes": int(len(le.classes_)),
            "majority_class_fraction": float(counts.iloc[0] / counts.sum()),
            "accuracy": accuracy_score(y_test, pred),
            "balanced_accuracy": balanced_accuracy_score(y_test, pred),
            "macro_f1": f1_score(y_test, pred, average="macro", zero_division=0),
        })

out = pd.DataFrame(rows)

if len(out) == 0:
    print("No valid confounder targets found.")
else:
    out = out.sort_values(["dataset", "balanced_accuracy"], ascending=[True, False])
    out.to_csv(OUT / "geneformer_confounder_predictability.csv", index=False)
    print("\nConfounder predictability:")
    print(out.to_string(index=False))
    print("\nWrote:", OUT / "geneformer_confounder_predictability.csv")
