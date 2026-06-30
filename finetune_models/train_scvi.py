from pathlib import Path
import yaml
import scanpy as sc
import anndata as ad
import scvi

CONFIG = "finetune_models/configs/datasets.yaml"
OUT = Path("finetune_models/outputs/scvi")
OUT.mkdir(parents=True, exist_ok=True)

with open(CONFIG, "r") as f:
    cfg = yaml.safe_load(f)

label_key = cfg["label_key"]
batch_key = cfg["batch_key"]
celltype_key = cfg["celltype_key"]

print("Loading training data...")
train_adatas = [sc.read_h5ad(p) for p in cfg["train"]]
adata_train = ad.concat(train_adatas, join="inner")

print("Loading test data...")
test_adatas = [sc.read_h5ad(p) for p in cfg["test"]]
adata_test = ad.concat(test_adatas, join="inner")

print("Train:", adata_train)
print("Test:", adata_test)
print("Train datasets:")
print(adata_train.obs[batch_key].value_counts())
print("Train labels:")
print(adata_train.obs[label_key].value_counts())

scvi.model.SCVI.setup_anndata(
    adata_train,
    layer="counts_like",
    batch_key=batch_key,
    labels_key=label_key,
)

model = scvi.model.SCVI(
    adata_train,
    n_latent=30,
    n_layers=2,
    n_hidden=128,
)

print("Training scVI...")
model.train(max_epochs=100)

print("Saving model...")
model.save(OUT / "model", overwrite=True)

print("Saving train latent representation...")
adata_train.obsm["X_scVI_finetuned"] = model.get_latent_representation()
adata_train.write_h5ad(OUT / "train_scvi_finetuned.h5ad")

print("Mapping test data into trained scVI model...")
scvi.model.SCVI.prepare_query_anndata(adata_test, model)
query_model = scvi.model.SCVI.load_query_data(adata_test, model)
query_model.train(max_epochs=50, plan_kwargs={"weight_decay": 0.0})

adata_test.obsm["X_scVI_finetuned"] = query_model.get_latent_representation()
adata_test.write_h5ad(OUT / "test_scvi_finetuned.h5ad")

query_model.save(OUT / "query_model", overwrite=True)

print("Done.")
print("Outputs saved to:", OUT)