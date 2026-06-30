from pathlib import Path
import pandas as pd
import anndata as ad
import tarfile
import gzip

ROOT = Path.cwd()

OUT_DIR = ROOT / "results_final6" / "inspection"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RAW = ROOT / "data" / "raw"

PATHS = {
    "mitsialis": RAW / "ibd_mitsialis_2020",
    "martin": RAW / "ibd_martin_2019",
    "oliver": RAW / "ibd_oliver_2024_cellxgene" / "manual_download",
}

DISEASE_KEYWORDS = [
    "disease", "diagnosis", "condition", "health", "status",
    "inflamed", "inflammation", "crohn", "uc", "colitis",
    "sample", "donor", "subject", "patient", "tissue", "source",
    "study", "dataset", "assay", "cell_type"
]


def add(lines, text=""):
    lines.append(str(text))


def inspect_h5ad(path, lines, max_values=30):
    add(lines, "=" * 120)
    add(lines, f"H5AD: {path}")
    try:
        a = ad.read_h5ad(path, backed="r")
        add(lines, f"Shape: {a.n_obs:,} cells x {a.n_vars:,} genes")

        add(lines, "\nOBS COLUMNS:")
        add(lines, ", ".join(list(a.obs.columns)))

        add(lines, "\nVAR COLUMNS:")
        add(lines, ", ".join(list(a.var.columns)))

        add(lines, "\nPOSSIBLE LABEL / METADATA COLUMNS:")
        for col in a.obs.columns:
            col_lower = str(col).lower()
            if any(k in col_lower for k in DISEASE_KEYWORDS):
                add(lines, f"\n--- {col} ---")
                try:
                    vc = a.obs[col].astype(str).value_counts(dropna=False).head(max_values)
                    add(lines, vc.to_string())
                except Exception as e:
                    add(lines, f"Could not value_count {col}: {e}")

        add(lines, "\nFIRST 20 VAR NAMES:")
        add(lines, ", ".join([str(x) for x in list(a.var_names[:20])]))

        for possible in ["feature_name", "gene_symbols", "gene_symbol", "ensembl_id", "gene_ids"]:
            if possible in a.var.columns:
                add(lines, f"\nFIRST 20 {possible}:")
                add(lines, ", ".join(a.var[possible].astype(str).head(20).tolist()))

        try:
            a.file.close()
        except Exception:
            pass

    except Exception as e:
        add(lines, f"ERROR reading h5ad: {repr(e)}")


def inspect_csv_gz(path, lines):
    add(lines, "=" * 120)
    add(lines, f"CSV/GZ: {path}")
    try:
        df = pd.read_csv(path, compression="gzip", nrows=5)
        add(lines, f"First 5 rows shape: {df.shape}")
        add(lines, "Columns:")
        add(lines, ", ".join(list(df.columns[:50])))
        add(lines, "\nHead:")
        add(lines, df.head().to_string())
    except Exception as e:
        add(lines, f"ERROR reading csv.gz: {repr(e)}")


def inspect_tar(path, lines):
    add(lines, "=" * 120)
    add(lines, f"TAR: {path}")
    try:
        with tarfile.open(path, "r") as t:
            names = t.getnames()
        add(lines, f"Files in tar: {len(names)}")
        add(lines, "\nFirst 100 files:")
        for n in names[:100]:
            add(lines, n)
    except Exception as e:
        add(lines, f"ERROR reading tar: {repr(e)}")


def inspect_folder(name, folder):
    lines = []
    add(lines, f"INSPECTION REPORT: {name}")
    add(lines, f"Folder: {folder}")
    add(lines, f"Exists: {folder.exists()}")

    if not folder.exists():
        out = OUT_DIR / f"{name}_inspection.txt"
        out.write_text("\n".join(lines), encoding="utf-8")
        return out

    files = sorted([p for p in folder.rglob("*") if p.is_file()])
    add(lines, f"\nTotal files found: {len(files)}")

    add(lines, "\nAll files:")
    for p in files[:300]:
        add(lines, str(p.relative_to(folder)))

    h5ads = [p for p in files if p.suffix.lower() == ".h5ad"]
    csvgzs = [p for p in files if p.name.lower().endswith(".csv.gz")]
    tars = [p for p in files if p.suffix.lower() == ".tar"]

    add(lines, f"\nH5AD files: {len(h5ads)}")
    add(lines, f"CSV.GZ files: {len(csvgzs)}")
    add(lines, f"TAR files: {len(tars)}")

    for p in h5ads:
        inspect_h5ad(p, lines)

    for p in csvgzs:
        inspect_csv_gz(p, lines)

    for p in tars:
        inspect_tar(p, lines)

    out = OUT_DIR / f"{name}_inspection.txt"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def main():
    outputs = []
    for name, folder in PATHS.items():
        out = inspect_folder(name, folder)
        outputs.append(out)

    print("Saved inspection reports:")
    for o in outputs:
        print(o)


if __name__ == "__main__":
    main()