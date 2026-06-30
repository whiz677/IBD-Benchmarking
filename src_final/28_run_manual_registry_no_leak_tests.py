from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    balanced_accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix,
)
from sklearn.model_selection import GroupKFold

ROOT = Path.cwd()
REG = ROOT / "results_final6/proper_no_leak_benchmark/manual_embedding_registry.csv"
OUT = ROOT / "results_final6/proper_no_leak_benchmark"
PRED_OUT = OUT / "manual_registry_predictions"
PRED_OUT.mkdir(parents=True, exist_ok=True)

def load_pair(emb, obs):
    X = np.load(emb)
    odf = pd.read_csv(obs)

    for c in ["Unnamed: 0", "index"]:
        if c in odf.columns:
            odf = odf.drop(columns=[c])

    if X.shape[0] != len(odf):
        raise ValueError(f"row mismatch {emb}: {X.shape} vs {odf.shape}")

    y = odf["y_ibd"].astype(int).values
    return X, y, odf

def make_clf():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(
            max_iter=5000,
            class_weight="balanced",
            solver="lbfgs",
            n_jobs=-1,
            random_state=42,
        )),
    ])

def metrics(y, prob, pred):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "n": int(len(y)),
        "positive_n": int((y == 1).sum()),
        "negative_n": int((y == 0).sum()),
        "auroc": float(roc_auc_score(y, prob)) if len(np.unique(y)) == 2 else None,
        "auprc": float(average_precision_score(y, prob)) if len(np.unique(y)) == 2 else None,
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }

def find_group_col(obs):
    for c in ["donor_label", "source_id", "sample_label", "patient_id", "sample_id"]:
        if c in obs.columns and obs[c].notna().sum() > 0 and obs[c].nunique() >= 3:
            return c
    return None

reg = pd.read_csv(REG)
rows = []

# 1. Locked external tests: train_dev -> Martin / Oliver
for (model, group), sub in reg.groupby(["model", "group"]):
    have = {r["split"]: r for _, r in sub.iterrows()}

    if "train_dev" not in have:
        continue

    X_train, y_train, obs_train = load_pair(have["train_dev"]["embedding_file"], have["train_dev"]["obs_file"])

    clf = make_clf()
    print(f"\nLOCKED EXTERNAL | {model} | {group} | train_dev {X_train.shape}", flush=True)
    clf.fit(X_train, y_train)

    for split in ["martin", "oliver_no_martin"]:
        if split not in have:
            continue

        X_test, y_test, obs_test = load_pair(have[split]["embedding_file"], have[split]["obs_file"])

        prob = clf.predict_proba(X_test)[:, 1]
        pred = (prob >= 0.5).astype(int)

        m = metrics(y_test, prob, pred)
        m.update({
            "test_type": "locked_external",
            "model": model,
            "group": group,
            "train_dataset": "train_dev",
            "test_dataset": split,
            "leakage_control": "train_on_3_source_train_dev_test_locked_external_no_random_split",
        })
        rows.append(m)

        pred_df = obs_test.copy()
        pred_df["prob_ibd"] = prob
        pred_df["pred_ibd"] = pred
        pred_df.to_csv(PRED_OUT / f"{model}_{group}_{split}_predictions.csv", index=False)

# 2. Donor/sample-blocked CV inside train_dev
for (model, group), sub in reg[reg["split"] == "train_dev"].groupby(["model", "group"]):
    r = sub.iloc[0]
    X, y, obs = load_pair(r["embedding_file"], r["obs_file"])

    group_col = find_group_col(obs)
    if group_col is None:
        print(f"\nSkipping donor-blocked CV for {model} {group}: no donor/sample column")
        continue

    groups = obs[group_col].astype(str).values
    n_groups = pd.Series(groups).nunique()
    n_splits = min(5, n_groups)

    print(f"\nDONOR-BLOCKED CV | {model} | {group} | group_col={group_col} | groups={n_groups}", flush=True)

    probs = np.zeros(len(y), dtype=float)
    preds = np.zeros(len(y), dtype=int)
    tested = np.zeros(len(y), dtype=bool)

    gkf = GroupKFold(n_splits=n_splits)

    for fold, (tr, te) in enumerate(gkf.split(X, y, groups), start=1):
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            print(f"  skipping fold {fold}; one class missing")
            continue

        clf = make_clf()
        print(f"  fold {fold}/{n_splits}: train={len(tr)}, heldout={len(te)}", flush=True)
        clf.fit(X[tr], y[tr])

        prob = clf.predict_proba(X[te])[:, 1]
        pred = (prob >= 0.5).astype(int)

        probs[te] = prob
        preds[te] = pred
        tested[te] = True

    if tested.sum() > 0:
        m = metrics(y[tested], probs[tested], preds[tested])
        m.update({
            "test_type": "donor_blocked_cv_oof",
            "model": model,
            "group": group,
            "train_dataset": "train_dev",
            "test_dataset": f"held_out_{group_col}",
            "group_col": group_col,
            "leakage_control": "donor_or_sample_group_never_in_both_train_and_test",
        })
        rows.append(m)

# 3. Source-held-out inside train_dev: train on 2 sources, test on 1 source
for (model, group), sub in reg[reg["split"] == "train_dev"].groupby(["model", "group"]):
    r = sub.iloc[0]
    X, y, obs = load_pair(r["embedding_file"], r["obs_file"])

    source_col = None
    for c in ["dataset_id", "dataset_eval"]:
        if c in obs.columns and obs[c].nunique() >= 3:
            source_col = c
            break

    if source_col is None:
        print(f"\nSkipping source-heldout for {model} {group}: no 3-source column")
        continue

    sources = sorted(obs[source_col].astype(str).unique())
    print(f"\nSOURCE-HELDOUT | {model} | {group} | source_col={source_col} | sources={sources}", flush=True)

    for held_source in sources:
        test_mask = obs[source_col].astype(str).eq(held_source).values
        train_mask = ~test_mask

        if len(np.unique(y[test_mask])) < 2 or len(np.unique(y[train_mask])) < 2:
            print("  skipping", held_source, "because one disease class missing")
            continue

        clf = make_clf()
        print(f"  hold out {held_source}: train={train_mask.sum()}, test={test_mask.sum()}", flush=True)
        clf.fit(X[train_mask], y[train_mask])

        prob = clf.predict_proba(X[test_mask])[:, 1]
        pred = (prob >= 0.5).astype(int)

        m = metrics(y[test_mask], prob, pred)
        m.update({
            "test_type": "source_heldout",
            "model": model,
            "group": group,
            "train_dataset": f"train_dev_except_{held_source}",
            "test_dataset": held_source,
            "source_col": source_col,
            "leakage_control": "entire_source_dataset_held_out_from_training",
        })
        rows.append(m)

out = pd.DataFrame(rows)
out_path = OUT / "manual_registry_no_leak_disease_metrics.csv"
out.to_csv(out_path, index=False)

print("\nDONE no-leak disease benchmark.")
print(out.to_string(index=False))
print("\nWrote:", out_path)
