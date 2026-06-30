# Dataset Inclusion and Exclusion Criteria

## Include a dataset if it has
1. Human single-cell RNA-seq data.
2. IBD/healthy or disease/control labels.
3. Donor or sample identifiers.
4. Cell-type annotations.
5. Dataset or study identifiers.
6. Assay/platform metadata if available.
7. Enough cells in both disease and control groups.
8. Enough cells in major cell types for cell-type-matched testing.

## Exclude or flag a dataset if it has
1. No donor/sample identifier.
2. No control group.
3. Ambiguous disease labels.
4. Missing cell-type labels.
5. Too few cells after filtering.
6. Perfect confounding between disease/control and dataset with no possible held-out test.

## Why this matters
The benchmark is only meaningful if the dataset can support leakage-resistant splits and confounder analysis.
