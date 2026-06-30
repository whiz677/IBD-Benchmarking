from pathlib import Path
import json
import yaml
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import scvi

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
OUT = Path("finetune_models/outputs/scvi_locked_external")
OUT.mkdir(parents=True, exist_ok=True)

with open(CONFIG, "r") as f:
    cfg = yaml.safe_load(f)

label_key = cfg["label_key"]
batch_key = cfg["batch_key"]

print("Loading train data...")
train_adatas = [sc.read_h5ad(p) for p in cfg["train"]]
adata_train = ad.concat(train_adatas, join="inner", index_unique="-train")

print("Loading locked external test data...")
test_adatas = []
for p in cfg["test"]:
    x = sc.read_h5ad(p)
    x.obs["test_file"] = Path(p).stem
    test_adatas.append(x)

adata_test = ad.concat(test_adatas, join="inner", index_unique="-test")

common_genes = adata_train.var_names.intersection(adata_test.var_names)
adata_train = adata_train[:, common_genes].copy()
adata_test = adata_test[:, common_genes].copy()

print("Train:", adata_train)
print("Test:", adata_test)
print("Shared genes:", len(common_genes))

print("Train datasets:")
print(adata_train.obs[batch_key].value_counts())
print("Test datasets:")
print(adata_test.obs[batch_key].value_counts())

print("Train labels:")
print(adata_train.obs[label_key].value_counts())
print("Test labels:")
print(adata_test.obs[label_key].value_counts())

print("Setting up scVI...")
scvi.model.SCVI.setup_anndata(
    adata_train,
    batch_key=batch_key,
    labels_key=label_key,
)

model = scvi.model.SCVI(
    adata_train,
    n_latent=30,
    n_layers=2,
    n_hidden=128,
)

print("Training scVI on Garrido-Trigo + Kong + Smillie...")
model.train(max_epochs=100)

print("Saving trained scVI model...")
model.save(OUT / "model", overwrite=True)

print("Getting train latent representation...")
X_train = model.get_latent_representation()
adata_train.obsm["X_scVI_finetuned"] = X_train
adata_train.write_h5ad(OUT / "train_scvi_finetuned.h5ad")

print("Mapping locked test data into trained scVI model...")
scvi.model.SCVI.prepare_query_anndata(adata_test, model)
query_model = scvi.model.SCVI.load_query_data(adata_test, model)
query_model.train(max_epochs=50, plan_kwargs={"weight_decay": 0.0})

X_test = query_model.get_latent_representation()
adata_test.obsm["X_scVI_finetuned"] = X_test
adata_test.write_h5ad(OUT / "test_scvi_finetuned.h5ad")
query_model.save(OUT / "query_model", overwrite=True)

print("Training comparable IBD classifier...")
y_train_raw = adata_train.obs[label_key].astype(str).values
y_test_raw = adata_test.obs[label_key].astype(str).values

label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train_raw)
y_test = label_encoder.transform(y_test_raw)
labels = label_encoder.classes_.tolist()

clf = LogisticRegression(
    max_iter=5000,
    class_weight="balanced",
    solver="lbfgs",
)
clf.fit(X_train, y_train)

pred = clf.predict(X_test)
proba = clf.predict_proba(X_test)

def compute_metrics(y_true, y_pred, y_prob, name):
    result = {
        "name": name,
        "labels": labels,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            target_names=labels,
            output_dict=True,
        ),
    }

    if len(labels) == 2:
        positive_index = 1
        result["positive_class"] = labels[positive_index]
        result["auroc"] = float(roc_auc_score(y_true, y_prob[:, positive_index]))
        result["auprc"] = float(
            average_precision_score(y_true, y_prob[:, positive_index])
        )
    else:
        result["auroc_macro_ovr"] = float(
            roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
        )
        result["auprc_macro_ovr"] = float(
            average_precision_score(y_true, y_prob, average="macro")
        )

    return result

metrics = {
    "overall": compute_metrics(y_test, pred, proba, "overall_locked_external"),
    "by_test_file": {},
    "by_dataset_eval": {},
}

for test_file in sorted(adata_test.obs["test_file"].astype(str).unique()):
    mask = adata_test.obs["test_file"].astype(str).values == test_file
    metrics["by_test_file"][test_file] = compute_metrics(
        y_test[mask],
        pred[mask],
        proba[mask],
        test_file,
    )

for dataset_name in sorted(adata_test.obs[batch_key].astype(str).unique()):
    mask = adata_test.obs[batch_key].astype(str).values == dataset_name
    metrics["by_dataset_eval"][dataset_name] = compute_metrics(
        y_test[mask],
        pred[mask],
        proba[mask],
        dataset_name,
    )

pred_df = pd.DataFrame({
    "true_label": y_test_raw,
    "pred_label": label_encoder.inverse_transform(pred),
    batch_key: adata_test.obs[batch_key].astype(str).values,
    "test_file": adata_test.obs["test_file"].astype(str).values,
})

for i, label in enumerate(labels):
    pred_df[f"prob_{label}"] = proba[:, i]

pred_df.to_csv(OUT / "ibd_test_predictions.csv", index=False)

with open(OUT / "ibd_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("Metrics:")
print(json.dumps(metrics, indent=2))
print("Saved outputs to:", OUT)