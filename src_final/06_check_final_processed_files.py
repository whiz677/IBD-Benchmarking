from pathlib import Path
import pandas as pd
import anndata as ad

ROOT = Path.cwd()

OUT_DIR = ROOT / "results_final6" / "readiness"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FILES = {
    "garrido_raw": ROOT / "data" / "raw" / "cellxgene" / "ibd_garrido_trigo" / "manual_download" / "cellxgene_download.h5ad",
    "kong_processed": ROOT / "data" / "processed" / "ibd_kong_2023_processed.h5ad",
    "smillie_epi_processed": ROOT / "data" / "processed" / "ibd_smillie_uc_processed.h5ad",
    "martin_locked_test": ROOT / "data" / "processed_final6" / "martin_locked_test_final.h5ad",
    "oliver_no_martin_locked_test": ROOT / "data" / "processed_final6" / "oliver_no_martin_locked_test_final.h5ad",
}

OUT_CSV = OUT_DIR / "final_processed_file_readiness.csv"
OUT_TXT = OUT_DIR / "final_processed_file_readiness.txt"


def find_col(obs, candidates):
    for c in candidates:
        if c in obs.columns:
            return c
    return None


def infer_y_ibd(obs):
    if "y_ibd" in obs.columns:
        y = pd.to_numeric(obs["y_ibd"], errors="coerce")
        return y

    disease_col = find_col(obs, ["disease", "donor_disease", "control_vs_disease"])
    if disease_col is None:
        return None

    def map_label(x):
        s = str(x).lower()
        if "normal" in s or "control" in s or "healthy" in s:
            return 0
        if "crohn" in s or "ulcerative" in s or "colitis" in s or "ibd" in s:
            return 1
        return None

    return obs[disease_col].map(map_label)


def broad_group_counts(obs):
    if "broad_cell_group" in obs.columns:
        return obs["broad_cell_group"].astype(str).value_counts().to_dict()
    if "cell_type_label" in obs.columns:
        return obs["cell_type_label"].astype(str).value_counts().head(10).to_dict()
    if "cell_type" in obs.columns:
        return obs["cell_type"].astype(str).value_counts().head(10).to_dict()
    return {}


def main():
    rows = []
    lines = []

    for dataset_id, path in FILES.items():
        lines.append("=" * 100)
        lines.append(dataset_id)
        lines.append(str(path))
        lines.append(f"exists: {path.exists()}")

        row = {
            "dataset_id": dataset_id,
            "path": str(path),
            "exists": path.exists(),
            "n_cells": None,
            "n_genes": None,
            "has_y_ibd": False,
            "n_class_0": None,
            "n_class_1": None,
            "has_both_classes": False,
            "donor_col": None,
            "n_donors": None,
            "celltype_col": None,
            "has_ensembl_ids": False,
            "usable_for_binary_ml": False,
            "notes": "",
        }

        if not path.exists():
            row["notes"] = "missing file"
            rows.append(row)
            lines.append("MISSING")
            continue

        try:
            a = ad.read_h5ad(path, backed="r")
            row["n_cells"] = a.n_obs
            row["n_genes"] = a.n_vars

            obs = a.obs.copy()
            var = a.var.copy()

            y = infer_y_ibd(obs)
            if y is not None:
                row["has_y_ibd"] = True
                counts = y.value_counts(dropna=True).to_dict()
                row["n_class_0"] = int(counts.get(0, 0))
                row["n_class_1"] = int(counts.get(1, 0))
                row["has_both_classes"] = row["n_class_0"] > 0 and row["n_class_1"] > 0

            donor_col = find_col(obs, ["donor_label", "donor_id", "donorID_unified", "Subject", "subject"])
            row["donor_col"] = donor_col
            if donor_col is not None:
                row["n_donors"] = obs[donor_col].astype(str).nunique()

            celltype_col = find_col(obs, ["broad_cell_group", "cell_type_label", "cell_type", "CellType"])
            row["celltype_col"] = celltype_col

            # Geneformer readiness.
            var_names_are_ensembl = sum(str(x).startswith("ENSG") for x in a.var_names[: min(1000, a.n_vars)]) > 100
            has_ensembl_col = "ensembl_id" in var.columns
            row["has_ensembl_ids"] = bool(var_names_are_ensembl or has_ensembl_col)

            row["usable_for_binary_ml"] = bool(
                row["has_both_classes"]
                and row["n_class_0"] >= 100
                and row["n_class_1"] >= 100
                and row["n_donors"] is not None
                and row["n_donors"] >= 2
            )

            if not row["usable_for_binary_ml"]:
                row["notes"] = "not usable yet for binary ML; inspect labels/donors"

            lines.append(f"shape: {a.n_obs:,} cells x {a.n_vars:,} genes")
            lines.append(f"obs columns: {list(obs.columns)}")
            lines.append(f"var columns: {list(var.columns)}")
            lines.append(f"y counts: 0={row['n_class_0']} 1={row['n_class_1']}")
            lines.append(f"donor col: {row['donor_col']} n_donors={row['n_donors']}")
            lines.append(f"celltype col: {row['celltype_col']}")
            lines.append(f"broad/cell group counts: {broad_group_counts(obs)}")
            lines.append(f"has Ensembl IDs: {row['has_ensembl_ids']}")
            lines.append(f"usable_for_binary_ml: {row['usable_for_binary_ml']}")

            try:
                a.file.close()
            except Exception:
                pass

        except Exception as e:
            row["notes"] = f"ERROR: {repr(e)}"
            lines.append(f"ERROR: {repr(e)}")

        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    OUT_TXT.write_text("\n".join(lines), encoding="utf-8")

    print("Saved:")
    print(OUT_CSV)
    print(OUT_TXT)

    print("\nSummary:")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()