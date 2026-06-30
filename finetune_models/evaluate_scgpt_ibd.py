from pathlib import Path
import json
import yaml
import numpy as np
import pandas as pd
import scanpy as sc
import torch

from scipy.sparse import issparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from scgpt.model import TransformerModel
from scgpt.preprocess import Preprocessor
from scgpt.tokenizer import tokenize_and_pad_batch
from scgpt.tokenizer.gene_tokenizer import GeneVocab
from scgpt.utils import set_seed

CONFIG = "finetune_models/configs/datasets.yaml"
MODEL_ROOT = Path("pretrained/scgpt")
MODEL_DIR = Path("pretrained/scgpt/whole-human")
OUT = Path("finetune_models/outputs/scgpt")
OUT.mkdir(parents=True, exist_ok=True)

FINE_TUNED_MODEL = OUT / "best_model.pt"

seed = 42
batch_size = 16
n_bins = 51
max_seq_len = 1201

pad_token = "<pad>"
special_tokens = [pad_token, "<cls>", "<eoc>"]
pad_value = -2
input_layer_key = "X_binned"

set_seed(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

if not FINE_TUNED_MODEL.exists():
    raise FileNotFoundError(
        f"Missing fine-tuned model: {FINE_TUNED_MODEL}. "
        "Run train_eval_scgpt.py first, or check whether training saved best_model.pt."
    )

with open(CONFIG, "r") as f:
    cfg = yaml.safe_load(f)

label_key = cfg["label_key"]
batch_key = cfg["batch_key"]

if not (MODEL_DIR / "vocab.json").exists():
    candidates = list(MODEL_ROOT.rglob("vocab.json"))
    if not candidates:
        raise FileNotFoundError("Could not find vocab.json under pretrained/scgpt")
    MODEL_DIR = candidates[0].parent

print("Using scGPT checkpoint config:", MODEL_DIR)
print("Using fine-tuned weights:", FINE_TUNED_MODEL)

vocab = GeneVocab.from_file(MODEL_DIR / "vocab.json")
for token in special_tokens:
    if token not in vocab:
        vocab.append_token(token)
vocab.set_default_index(vocab[pad_token])

with open(MODEL_DIR / "args.json", "r") as f:
    model_configs = json.load(f)

embsize = model_configs["embsize"]
nhead = model_configs["nheads"]
d_hid = model_configs["d_hid"]
nlayers = model_configs["nlayers"]
n_layers_cls = model_configs.get("n_layers_cls", 3)

print("Loading data...")
adata_train = sc.read_h5ad(cfg["train"][0])
adata_test = sc.read_h5ad(cfg["test"][0])

adata_train.var["gene_name"] = adata_train.var.index.astype(str)
adata_test.var["gene_name"] = adata_test.var.index.astype(str)

adata_train.var["id_in_vocab"] = [
    1 if gene in vocab else -1 for gene in adata_train.var["gene_name"]
]
adata_test.var["id_in_vocab"] = [
    1 if gene in vocab else -1 for gene in adata_test.var["gene_name"]
]

adata_train = adata_train[:, adata_train.var["id_in_vocab"] >= 0].copy()
adata_test = adata_test[:, adata_test.var["id_in_vocab"] >= 0].copy()

common_genes = adata_train.var_names.intersection(adata_test.var_names)
adata_train = adata_train[:, common_genes].copy()
adata_test = adata_test[:, common_genes].copy()

print("Train:", adata_train)
print("Test:", adata_test)
print("Matched genes:", len(common_genes))
print("Train datasets:")
print(adata_train.obs[batch_key].value_counts())
print("Train labels:")
print(adata_train.obs[label_key].value_counts())
print("Test labels:")
print(adata_test.obs[label_key].value_counts())

adata_train.obs["str_batch"] = adata_train.obs[batch_key].astype(str)
adata_test.obs["str_batch"] = adata_test.obs[batch_key].astype(str)

all_batches = sorted(
    set(adata_train.obs["str_batch"]).union(set(adata_test.obs["str_batch"]))
)
batch_map = {batch: i for i, batch in enumerate(all_batches)}
adata_train.obs["batch_id"] = adata_train.obs["str_batch"].map(batch_map).astype(int)
adata_test.obs["batch_id"] = adata_test.obs["str_batch"].map(batch_map).astype(int)

preprocessor = Preprocessor(
    use_key="counts_like",
    filter_gene_by_counts=False,
    filter_cell_by_counts=False,
    normalize_total=1e4,
    result_normed_key="X_normed",
    log1p=True,
    result_log1p_key="X_log1p",
    subset_hvg=False,
    binning=n_bins,
    result_binned_key=input_layer_key,
)

print("Preprocessing train...")
preprocessor(adata_train, batch_key="str_batch")
print("Preprocessing test...")
preprocessor(adata_test, batch_key="str_batch")

genes = adata_train.var["gene_name"].tolist()
gene_ids = np.array(vocab(genes), dtype=int)


def matrix_from_layer(adata, key):
    x = adata.layers[key]
    return x.toarray() if issparse(x) else np.asarray(x)


print("Tokenizing train/test...")
train_counts = matrix_from_layer(adata_train, input_layer_key)
test_counts = matrix_from_layer(adata_test, input_layer_key)

tokenized_train = tokenize_and_pad_batch(
    train_counts,
    gene_ids,
    max_len=max_seq_len,
    vocab=vocab,
    pad_token=pad_token,
    pad_value=pad_value,
    append_cls=True,
    include_zero_gene=True,
)

tokenized_test = tokenize_and_pad_batch(
    test_counts,
    gene_ids,
    max_len=max_seq_len,
    vocab=vocab,
    pad_token=pad_token,
    pad_value=pad_value,
    append_cls=True,
    include_zero_gene=True,
)

model = TransformerModel(
    len(vocab),
    embsize,
    nhead,
    d_hid,
    nlayers,
    nlayers_cls=n_layers_cls,
    vocab=vocab,
    dropout=0.2,
    pad_token=pad_token,
    pad_value=pad_value,
    do_mvc=False,
    do_dab=False,
    use_batch_labels=False,
    num_batch_labels=len(all_batches),
    domain_spec_batchnorm=False,
    n_input_bins=n_bins,
    ecs_threshold=0.0,
    explicit_zero_prob=True,
    use_fast_transformer=False,
    pre_norm=False,
)

print("Loading fine-tuned scGPT weights...")
state = torch.load(FINE_TUNED_MODEL, map_location="cpu")
model.load_state_dict(state)
model.to(device)
model.eval()


def encode_cls(tokenized, name):
    gene_ids_all = tokenized["genes"]
    values_all = tokenized["values"].float()
    mask_all = gene_ids_all.eq(vocab[pad_token])

    chunks = []
    n = gene_ids_all.shape[0]

    with torch.no_grad():
        for start in tqdm(range(0, n, batch_size), desc=f"encoding {name}"):
            end = min(start + batch_size, n)

            src = gene_ids_all[start:end].to(device)
            values = values_all[start:end].to(device)
            mask = mask_all[start:end].to(device)

            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                output = model(
                    src,
                    values,
                    src_key_padding_mask=mask,
                    MVC=False,
                    ECS=False,
                )
                emb = output["cell_emb"]

            chunks.append(emb.detach().cpu().numpy())

    x = np.concatenate(chunks, axis=0)
    x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    return x


print("Encoding train/test as CLS cell embeddings...")
X_train = encode_cls(tokenized_train, "train")
X_test = encode_cls(tokenized_test, "test")

adata_train.obsm["X_scGPT_finetuned"] = X_train
adata_test.obsm["X_scGPT_finetuned"] = X_test

print("Saving scGPT embeddings...")
adata_train.write_h5ad(OUT / "train_scgpt_finetuned.h5ad")
adata_test.write_h5ad(OUT / "test_scgpt_finetuned.h5ad")

print("Training comparable IBD classifier...")
y_train_raw = adata_train.obs[label_key].astype(str).values
y_test_raw = adata_test.obs[label_key].astype(str).values

label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train_raw)
y_test = label_encoder.transform(y_test_raw)
labels = label_encoder.classes_.tolist()

clf = LogisticRegression(max_iter=5000, class_weight="balanced", solver="lbfgs")
clf.fit(X_train, y_train)

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
    metrics["auprc"] = float(
        average_precision_score(y_test, proba[:, positive_index])
    )
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
print(OUT / "train_scgpt_finetuned.h5ad")
print(OUT / "test_scgpt_finetuned.h5ad")
print(OUT / "ibd_metrics.json")
print(OUT / "ibd_test_predictions.csv")