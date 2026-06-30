# Windows no-Census setup

Use this path if `pip install cellxgene-census` fails on Windows because of `tiledbsoma`.

## 1. Install packages without Census

```bat
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements_windows_no_census.txt
```

## 2. Check environment

```bat
python src\00_check_environment.py
```

This Windows version intentionally skips `cellxgene_census`.

## 3. Manually download the IBD `.h5ad`

Go to the CELLxGENE collection page for:

`Human inflammatory bowel disease (IBD) and healthy control 10x Single-cell transcriptomics data`

Collection ID:

`7c7bd6c2-925b-4034-baab-620ef1b760e1`

Download the dataset in `.h5ad` format.

Put the downloaded `.h5ad` file here:

```text
data\raw\cellxgene\ibd_garrido_trigo\manual_download\
```

## 4. Prepare the downloaded file

```bat
python src\01_prepare_manual_h5ad.py
```

This creates the canonical raw data file expected by the rest of the pipeline:

```text
data\raw\cellxgene\ibd_garrido_trigo\ibd_garrido_trigo_raw_census_slice.h5ad
```

## 5. Continue the pipeline

```bat
python src\02_metadata_audit.py
python src\03_preprocess_ibd.py
python src\04_subsample_ibd.py
python src\05_make_splits.py
python src\06_make_baseline_embeddings.py
python src\07_train_disease_prediction.py
python src\08_train_confounder_prediction.py
python src\09_calculate_leakage_scores.py
python src\10_make_pseudobulk.py
python src\11_negative_controls.py
python src\12_pathway_validation.py
python src\13_make_figures.py
```
