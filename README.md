# IBD-Benchmarking
This repository contains the code for a single-cell RNA-seq benchmark of disease-transfer representations in inflammatory bowel disease (IBD). The project compares PCA, scVI, scGPT, and Geneformer embeddings for predicting IBD versus control status across intestinal single-cell datasets.

The main goal was to test whether single-cell representations capture transferable IBD-associated biology or whether performance is driven by donor, source-study, tissue, cell-type, or other non-disease structure.
## Repository Structure

```text
src_final/        Final manuscript-ready analysis scripts
docs/             Notes, summaries, and project documentation
figures/          Manuscript or analysis figures
results_final6/   Final small result tables and summaries
splits/           Train/test split files, if included
