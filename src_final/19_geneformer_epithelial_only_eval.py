from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)

ROOT = Path.cwd()
EMB_DIR = ROOT / "embeddings_final6" / "geneformer"
OUT_DIR = ROOT / "results_final6" / "geneformer_final"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = {
    "train_dev": {
        "emb": EMB_DIR / "train_dev_all_cells_common_geneformer_embeddings.npy",
        "obs": EMB_DIR / "train_dev_all_cells_common_geneformer_obs.csv",
    },
    "locked_test_martin": {
        "emb": EMB_DIR / "locked_test_martin_all_cells_common_geneformer_embeddings.npy",
        "obs": EMB_DIR / "locked_test_martin_all_cells_common_geneformer_obs.csv",
    },
    "locked_test_oliver_no_martin": {
        "emb": EMB_DIR / "locked_test_oliver_no_martin_all_cells_common_geneformer_embeddings.npy",
        "obs": EMB_DIR / "locked_test_oliver_no_martin_all_cells_common_geneformer_obs.csv",
    },
}

def load_epithelial(name):
    X = np.load(DATASETS[name]["emb"])
    obs = pd.read_csv(DATASETS[name]["obs"])

    for c in ["Unnamed: 0", "index"]:
        if c in obs.columns:
            obs = obs.drop(columns=[c])

    if X.shape[0] != len(obs):
        raise ValueError(f"Row mismatch for {name}: X={X.shape}, obs={obs.shape}")

    if "broad_cell_group" not in obs.columns:
        raise ValueError(f"Missing broad_cell_group in {name}")

    mask = obs["broad_cell_group"].astype(str).str.lower().eq("epithelial").values
    X = X[mask]
    obs = obs.loc[mask].reset_index(drop=True)
    y = obs["y_ibd"].astype(int).values

    return X, y, obs

def metrics(y, prob, pred):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "n": int(len(y)),
        "positive_n": int((y == 1).sum()),
        "negative_n": int((y == 0).sum()),
        "auroc": float(roc_auc_score(y, prob)),
        "auprc": float(average_precision_score(y, prob)),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

print("Loading epithelial-only Geneformer embeddings...")

X_train, y_train, obs_train = load_epithelial("train_dev")
X_martin, y_martin, obs_martin = load_epithelial("locked_test_martin")
X_oliver, y_oliver, obs_oliver = load_epithelial("locked_test_oliver_no_martin")

print("train epithelial:", X_train.shape, np.bincount(y_train))
print("martin epithelial:", X_martin.shape, np.bincount(y_martin))
print("oliver epithelial:", X_oliver.shape, np.bincount(y_oliver))

clf = Pipeline([
    ("scaler", StandardScaler()),
    ("logreg", LogisticRegression(
        max_iter=5000,
        class_weight="balanced",
        solver="lbfgs",
        n_jobs=-1,
        random_state=42,
    )),
])

clf.fit(X_train, y_train)

rows = []

for name, X, y, obs in [
    ("train_dev_internal_epithelial", X_train, y_train, obs_train),
    ("locked_test_martin_epithelial", X_martin, y_martin, obs_martin),
    ("locked_test_oliver_no_martin_epithelial", X_oliver, y_oliver, obs_oliver),
]:
    prob = clf.predict_proba(X)[:, 1]
    pred = (prob >= 0.5).astype(int)

    m = metrics(y, prob, pred)
    m["model"] = "Geneformer_frozen_embeddings_logistic_regression"
    m["group"] = "epithelial"
    m["dataset"] = name
    rows.append(m)

    pred_df = obs.copy()
    pred_df["geneformer_prob_ibd"] = prob
    pred_df["geneformer_pred_ibd"] = pred
    pred_df.to_csv(OUT_DIR / f"{name}_geneformer_predictions.csv", index=False)

df = pd.DataFrame(rows)
df.to_csv(OUT_DIR / "geneformer_epithelial_locked_test_metrics.csv", index=False)

print("\nGeneformer epithelial-only metrics:")
print(df.to_string(index=False))
print("\nWrote:", OUT_DIR / "geneformer_epithelial_locked_test_metrics.csv")
