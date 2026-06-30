from pathlib import Path
import json
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
    classification_report,
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

LABEL_COL = "y_ibd"

def load_dataset(name):
    emb_path = DATASETS[name]["emb"]
    obs_path = DATASETS[name]["obs"]

    if not emb_path.exists():
        raise FileNotFoundError(f"Missing embeddings: {emb_path}")
    if not obs_path.exists():
        raise FileNotFoundError(f"Missing obs: {obs_path}")

    X = np.load(emb_path)
    obs = pd.read_csv(obs_path)

    # Remove index columns created by CSV saving.
    for c in ["Unnamed: 0", "index"]:
        if c in obs.columns:
            obs = obs.drop(columns=[c])

    if LABEL_COL not in obs.columns:
        raise ValueError(f"{LABEL_COL} not found in {obs_path}. Columns: {list(obs.columns)}")

    y = obs[LABEL_COL].astype(int).values

    if X.shape[0] != len(obs):
        raise ValueError(f"Row mismatch for {name}: X={X.shape}, obs={obs.shape}")

    return X, y, obs

def metrics_dict(y_true, y_prob, y_pred):
    out = {}
    out["n"] = int(len(y_true))
    out["positive_n"] = int(np.sum(y_true == 1))
    out["negative_n"] = int(np.sum(y_true == 0))

    # Some test sets could theoretically contain one class only.
    if len(np.unique(y_true)) == 2:
        out["auroc"] = float(roc_auc_score(y_true, y_prob))
        out["auprc"] = float(average_precision_score(y_true, y_prob))
    else:
        out["auroc"] = None
        out["auprc"] = None

    out["accuracy"] = float(accuracy_score(y_true, y_pred))
    out["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))
    out["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
    out["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
    out["recall"] = float(recall_score(y_true, y_pred, zero_division=0))

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out["tn"] = int(tn)
    out["fp"] = int(fp)
    out["fn"] = int(fn)
    out["tp"] = int(tp)

    return out

print("Loading datasets...")
X_train, y_train, obs_train = load_dataset("train_dev")
X_martin, y_martin, obs_martin = load_dataset("locked_test_martin")
X_oliver, y_oliver, obs_oliver = load_dataset("locked_test_oliver_no_martin")

print("Train:", X_train.shape, np.bincount(y_train))
print("Martin:", X_martin.shape, np.bincount(y_martin))
print("Oliver no Martin:", X_oliver.shape, np.bincount(y_oliver))

print("\nTraining logistic regression on frozen Geneformer embeddings...")

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
    ("train_dev_internal", X_train, y_train, obs_train),
    ("locked_test_martin", X_martin, y_martin, obs_martin),
    ("locked_test_oliver_no_martin", X_oliver, y_oliver, obs_oliver),
]:
    print("\n" + "=" * 90)
    print(name)

    y_prob = clf.predict_proba(X)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    m = metrics_dict(y, y_prob, y_pred)
    m["model"] = "Geneformer_frozen_embeddings_logistic_regression"
    m["dataset"] = name
    rows.append(m)

    print(json.dumps(m, indent=2))
    print("\nClassification report:")
    print(classification_report(y, y_pred, digits=4, zero_division=0))

    pred_df = obs.copy()
    pred_df["geneformer_prob_ibd"] = y_prob
    pred_df["geneformer_pred_ibd"] = y_pred
    pred_df.to_csv(OUT_DIR / f"{name}_geneformer_predictions.csv", index=False)

results = pd.DataFrame(rows)
results.to_csv(OUT_DIR / "geneformer_locked_test_metrics.csv", index=False)

print("\nDONE.")
print("Wrote:")
print(" ", OUT_DIR / "geneformer_locked_test_metrics.csv")
for p in sorted(OUT_DIR.glob("*geneformer_predictions.csv")):
    print(" ", p)
