from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

outdir = Path("figures/geneformer_final")
outdir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv("results_final6/geneformer_final/geneformer_locked_test_metrics.csv")

plot_df = df[df["dataset"].isin([
    "locked_test_martin",
    "locked_test_oliver_no_martin",
])].copy()

for metric in ["auroc", "auprc", "balanced_accuracy", "f1"]:
    plt.figure(figsize=(6, 4))
    plt.bar(plot_df["dataset"], plot_df[metric])
    plt.ylim(0, 1)
    plt.ylabel(metric.upper())
    plt.xticks(rotation=20, ha="right")
    plt.title(f"Geneformer frozen embeddings: {metric.upper()}")
    plt.tight_layout()
    path = outdir / f"geneformer_{metric}.png"
    plt.savefig(path, dpi=300)
    print("wrote", path)
