from pathlib import Path
import json
import numpy as np
import pandas as pd
import scanpy as sc
import yaml

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.preprocessing import LabelEncoder

CONFIG = "finetune_models/configs/datasets.yaml"
OUT = Path("finetune_models/outputs/scvi")

with open(CONFIG, "r") as f:
    cfg = yaml.safe_load(f)

label_key = cfg["label_key"]
batch_key = cfg["batch_key"]
embedding_key = "X_scVI_finetuned"

print("Loading scVI embedding outputs...")
adata_train = sc.read_h5ad(OUT / "train_scvi_finetuned.h5ad")
adata_test = sc.read_h5ad(OUT / "test_scvi_finetuned.h5ad")

print("Train:", adata_train)
print("Test:", adata_test)

if embedding_key not in adata_train.obsm:
    raise ValueError(f"Missing {embedding_key} in train obsm")
if embedding_key not in adata_test.obsm:
    raise ValueError(f"Missing {embedding_key} in test obsm")

print("Train label counts:")
print(adata_train.obs[label_key].value_counts())

print("Test label counts:")
print(adata_test.obs[label_key].value_counts())

X_train = adata_train.obsm[embedding_key]
X_test = adata_test.obsm[embedding_key]

y_train_raw = adata_train.obs[label_key].astype(str).values
y_test_raw = adata_test.obs[label_key].astype(str).values

label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train_raw)
y_test = label_encoder.transform(y_test_raw)

labels = label_encoder.classes_.tolist()
print("Labels:", labels)

clf = LogisticRegression(
    max_iter=5000,
    class_weight="balanced",
    solver="lbfgs",
)

print("Training IBD classifier on scVI embeddings...")
clf.fit(X_train, y_train)

print("Predicting test set...")
pred = clf.predict(X_test)
proba = clf.predict_proba(X_test)

metrics = {
    "labels": labels,
    "accuracy": float(accuracy_score(y_test, pred)),
    "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
    "classification_report": classification_report(
        y_test,
        pred,
        target_names=labels,
        output_dict=True,
    ),
}

if len(labels) == 2:
    positive_index = 1
    metrics["positive_class"] = labels[positive_index]
    metrics["auroc"] = float(roc_auc_score(y_test, proba[:, positive_index]))
    metrics["auprc"] = float(average_precision_score(y_test, proba[:, positive_index]))
else:
    metrics["auroc_macro_ovr"] = float(
        roc_auc_score(y_test, proba, multi_class="ovr", average="macro")
    )
    metrics["auprc_macro_ovr"] = float(
        average_precision_score(y_test, proba, average="macro")
    )

pred_df = pd.DataFrame({
    "true_label": y_test_raw,
    "pred_label": label_encoder.inverse_transform(pred),
    batch_key: adata_test.obs[batch_key].astype(str).values,
})

for i, label in enumerate(labels):
    pred_df[f"prob_{label}"] = proba[:, i]

pred_df.to_csv(OUT / "ibd_test_predictions.csv", index=False)

with open(OUT / "ibd_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("Metrics:")
print(json.dumps(metrics, indent=2))
print("Saved:")
print(OUT / "ibd_metrics.json")
print(OUT / "ibd_test_predictions.csv")