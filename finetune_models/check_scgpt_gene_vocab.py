import scanpy as sc
from scgpt.tokenizer.gene_tokenizer import GeneVocab

h5ad_path = "./data_fixed/combined_garrido_kong_smillie_hvg.fixed.h5ad"
vocab_path = "pretrained/scgpt/whole-human/vocab.json"

adata = sc.read_h5ad(h5ad_path)
vocab = GeneVocab.from_file(vocab_path)
vocab_genes = set(vocab.get_stoi().keys())

genes = adata.var_names.astype(str).tolist()

exact = [g for g in genes if g in vocab_genes]
upper = [g for g in genes if g.upper() in vocab_genes]
lower = [g for g in genes if g.lower() in vocab_genes]

print("Total genes in h5ad:", len(genes))
print("Exact matches:", len(exact))
print("Uppercase matches:", len(upper))
print("Lowercase matches:", len(lower))
print()
print("First 50 h5ad var_names:")
print(genes[:50])
print()
print("First 80 scGPT vocab genes:")
print(list(vocab.get_stoi().keys())[:80])
print()
print("adata.var columns:")
print(adata.var.columns.tolist())

for col in adata.var.columns:
    vals = adata.var[col].astype(str).tolist()
    matches = sum(v in vocab_genes for v in vals)
    if matches > 0:
        print(f"Column {col!r} matches scGPT vocab: {matches}/{len(vals)}")