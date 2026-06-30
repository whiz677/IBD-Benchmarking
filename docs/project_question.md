# Project Question

## Title
scDiseaseShift-IBD: A confounder-controlled benchmark for disease-state generalization in single-cell models using inflammatory bowel disease gut atlases.

## Main question
Do single-cell model embeddings encode transferable IBD gut disease biology, or do they primarily reflect cell type, donor identity, dataset source, assay/platform, tissue region, and batch structure?

## Main task
IBD vs healthy control.

## Secondary task
Ulcerative colitis vs Crohn's disease.

## Hypothesis
Naive random cell splits overestimate IBD disease-state prediction because cells from the same donor, dataset, assay, tissue region, and cell-type composition can leak across train and test sets.

## Main contribution
A reusable IBD/gut benchmark with metadata audit tables, leakage-resistant split files, pseudobulk validation, negative controls, and biological pathway validation.
