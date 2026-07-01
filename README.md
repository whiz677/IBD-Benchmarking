# IBD-Benchmarking
This repository contains the code for a single-cell RNA-seq benchmark of disease-transfer representations in inflammatory bowel disease (IBD). The project compares PCA, scVI, scGPT, and Geneformer embeddings for predicting IBD versus control status across intestinal single-cell datasets.

The main goal was to test whether single-cell representations capture transferable IBD-associated biology or whether performance is driven by donor, source-study, tissue, cell-type, or other non-disease structure.

Additional locked external fine-tuning scripts are provided in `finetune_models/`. Large fine-tuned model checkpoints, tokenized datasets, h5ad files, and output artifacts are not included in this repository because of file-size limits.
## Repository Structure
src_final/        Final manuscript-ready analysis scripts
docs/             Notes, summaries, and project documentation
figures/          Manuscript or analysis figures
results_final6/   Final small result tables and summaries
splits/           Train/test split files, if included

## Datasets
The benchmark used five public human intestinal single-cell RNA-seq datasets:
Garrido-Trigo dataset - training
Kong dataset - training
Smillie epithelial dataset - epithelial training
Martin dataset - locked external test set
Oliver-no-Martin integrated inflammatory gut atlas - locked external test set
Disease labels were standardized to a binary IBD-versus-control task. Crohn disease and ulcerative colitis samples were treated as IBD where available, and healthy or non-inflamed control labels were treated as control.
Large raw and processed data files are not stored in this repository because of file-size limits. Dataset sources and accessions are described in the manuscript and in the data-availability notes.
## Main Analysis
The final benchmark is implemented in:
src_final/
The final workflow includes:
dataset registry construction
dataset inspection and processing
locked external test construction
common-gene matrix construction
PCA and scVI embedding generation
scGPT frozen embedding generation
Geneformer frozen embedding generation
downstream logistic-regression disease prediction
locked external evaluation
donor-level evaluation
calibration analysis
paired model comparisons
confounder-predictability analysis
final result collection

## Metrics
Primary metrics:
AUROC
AUPRC
Additional metrics:
balanced accuracy
F1 score
confusion-matrix counts
donor-level AUROC and AUPRC
Brier score
expected calibration error
Uncertainty for locked external testing was estimated using donor-aware bootstrap resampling where donor identifiers were available.

## Installation
Create the environment using either Conda:
conda env create -f environment.yml
conda activate scdiseaseshift
or pip:
pip install -r requirements.txt
A Windows-specific requirements file is also included:
pip install -r requirements_windows_no_census.txt

## Running the Final Code
The final scripts are numbered in approximate execution order inside src_final/.
Example:
python src_final/01_make_final_dataset_registry.py
python src_final/06_check_final_processed_files.py
python src_final/07_build_final_train_test_matrices.py
python src_final/08_train_pca_scvi_final.py
python src_final/09_run_scgpt_final.py
python src_final/10A_make_geneformer_final_inputs.py
python src_final/10B_run_geneformer_final.py
python src_final/11_geneformer_downstream_eval.py
python src_final/12_collect_final_model_comparison.py
python src_final/28_run_manual_registry_no_leak_tests.py
Some scripts require large processed .h5ad files, pretrained model files, or saved embeddings that are not included in this repository which are available upon request.

## Reproducibility Notes
The final no-leak benchmark uses saved dataset roles, common-gene matrices, embedding files, and locked external labels. Random operations such as sampling, train/test splitting, and bootstrap resampling used fixed random seeds where applicable.
