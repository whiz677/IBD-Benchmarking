from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path.cwd()

IN_FILE = ROOT / "results" / "three_dataset_benchmark" / "three_dataset_benchmark_results.csv"
OUT_DIR = ROOT / "results" / "three_dataset_benchmark_summary"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    df = pd.read_csv(IN_FILE)

    ranked = df.sort_values("auroc", ascending=False)
    ranked.to_csv(OUT_DIR / "three_dataset_results_ranked.csv", index=False)

    # Main valid Smillie results: epithelial only.
    epithelial = df[df["broad_cell_group"] == "epithelial"].copy()
    epithelial = epithelial.sort_values("auroc", ascending=False)
    epithelial.to_csv(OUT_DIR / "three_dataset_epithelial_results_ranked.csv", index=False)

    # Leave-one-dataset-out results.
    lodo = df[df["analysis_type"].str.contains("lodo|leave_one", case=False, na=False)].copy()
    lodo = lodo.sort_values("auroc", ascending=False)
    lodo.to_csv(OUT_DIR / "three_dataset_lodo_results_ranked.csv", index=False)

    # Best model per condition.
    best = (
        df.sort_values("auroc", ascending=False)
        .groupby(["analysis_type", "broad_cell_group", "train_dataset", "test_dataset"])
        .head(1)
        .reset_index(drop=True)
    )
    best.to_csv(OUT_DIR / "best_model_per_three_dataset_condition.csv", index=False)

    # Model summary.
    summary = (
        df.groupby(["model", "analysis_type", "broad_cell_group"])
        .agg(
            n_tests=("auroc", "count"),
            mean_auroc=("auroc", "mean"),
            max_auroc=("auroc", "max"),
            mean_auprc=("auprc", "mean"),
            max_auprc=("auprc", "max"),
            mean_balanced_accuracy=("balanced_accuracy", "mean"),
        )
        .reset_index()
        .sort_values("max_auroc", ascending=False)
    )
    summary.to_csv(OUT_DIR / "three_dataset_model_summary.csv", index=False)

    # Top 20 plot.
    top = ranked.head(20).iloc[::-1].copy()
    top["label"] = (
        top["model"].astype(str)
        + " | "
        + top["analysis_type"].astype(str)
        + " | "
        + top["broad_cell_group"].astype(str)
        + " | "
        + top["train_dataset"].astype(str)
        + "→"
        + top["test_dataset"].astype(str)
    )

    plt.figure(figsize=(13, 8))
    plt.barh(range(len(top)), top["auroc"])
    plt.axvline(0.5, linestyle="--")
    plt.yticks(range(len(top)), top["label"])
    plt.xlabel("AUROC")
    plt.title("Top 3-dataset cross-atlas IBD results")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "figure_top20_three_dataset_auroc.png", dpi=300)
    plt.close()

    # Epithelial-only plot.
    epi_top = epithelial.head(15).iloc[::-1].copy()
    epi_top["label"] = (
        epi_top["model"].astype(str)
        + " | "
        + epi_top["analysis_type"].astype(str)
        + " | "
        + epi_top["train_dataset"].astype(str)
        + "→"
        + epi_top["test_dataset"].astype(str)
    )

    plt.figure(figsize=(12, 7))
    plt.barh(range(len(epi_top)), epi_top["auroc"])
    plt.axvline(0.5, linestyle="--")
    plt.yticks(range(len(epi_top)), epi_top["label"])
    plt.xlabel("AUROC")
    plt.title("Epithelial-matched 3-dataset transfer")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "figure_epithelial_three_dataset_auroc.png", dpi=300)
    plt.close()

    text = f"""Three-dataset benchmark summary

Datasets:
- Garrido-Trigo IBD/healthy colon
- Kong Crohn/normal ileum-colon
- Smillie UC/healthy epithelial colon subset

Important caveat:
Smillie is epithelial-only in this processed version, so the most valid third-dataset tests are epithelial-matched pairwise and leave-one-dataset-out evaluations.

Top overall result:
{ranked.iloc[0]['model']} | {ranked.iloc[0]['analysis_type']} | {ranked.iloc[0]['broad_cell_group']} | {ranked.iloc[0]['train_dataset']}→{ranked.iloc[0]['test_dataset']} | AUROC={ranked.iloc[0]['auroc']:.3f}

Top epithelial result:
{epithelial.iloc[0]['model']} | {epithelial.iloc[0]['analysis_type']} | {epithelial.iloc[0]['train_dataset']}→{epithelial.iloc[0]['test_dataset']} | AUROC={epithelial.iloc[0]['auroc']:.3f}

Top leave-one-dataset-out result:
{lodo.iloc[0]['model']} | {lodo.iloc[0]['analysis_type']} | {lodo.iloc[0]['broad_cell_group']} | {lodo.iloc[0]['train_dataset']}→{lodo.iloc[0]['test_dataset']} | AUROC={lodo.iloc[0]['auroc']:.3f}

Main interpretation:
Adding Smillie as a third independent UC epithelial dataset strengthens the benchmark. The strongest third-dataset evidence comes from epithelial-matched transfer and leave-one-dataset-out validation. scVI remains strong, especially for epithelial transfer, although PCA is competitive in some Smillie epithelial tests. This supports the broader conclusion that biologically matched evaluation is critical for cross-atlas IBD disease-state transfer.
"""

    (OUT_DIR / "three_dataset_summary.txt").write_text(text, encoding="utf-8")

    print("\nSaved outputs to:")
    print(OUT_DIR)
    print("\nOpen:")
    print(OUT_DIR / "three_dataset_summary.txt")
    print(OUT_DIR / "three_dataset_epithelial_results_ranked.csv")
    print(OUT_DIR / "three_dataset_lodo_results_ranked.csv")

if __name__ == "__main__":
    main()
