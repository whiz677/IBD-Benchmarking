# scDiseaseShift-IBD Automated Results Summary
This report was generated automatically from the output CSV files in the project folder.


# 1. Dataset audit
## Dataset summary file
|   n_cells |   n_genes | has_disease   | has_cell_type   | has_donor_id   | has_dataset_id   | has_assay   | has_tissue   |   n_donors |   n_diseases |   n_cell_types |   n_datasets |   n_assays |
|----------:|----------:|:--------------|:----------------|:---------------|:-----------------|:------------|:-------------|-----------:|-------------:|---------------:|-------------:|-----------:|
|     46700 |     32354 | True          | True            | True           | False            | True        | True         |         18 |            3 |              5 |          nan |          1 |
## Processed data overview
- Processed cells: **45784**

Disease binary label counts in processed data:
|   y_ibd |   n_cells |
|--------:|----------:|
|       1 |     30798 |
|       0 |     14986 |

Disease label counts:
| disease_label      |   n_cells |
|:-------------------|----------:|
| Crohn disease      |     16037 |
| normal             |     14986 |
| ulcerative colitis |     14761 |

Top cell types:
| cell_type_label                         |   n_cells |
|:----------------------------------------|----------:|
| plasma cell                             |     14949 |
| colon epithelial cell                   |     12219 |
| T cell of anorectum                     |     12120 |
| myeloid cell                            |      3676 |
| stromal cell of lamina propria of colon |      2820 |

- Number of donors: **18**

## Balanced subset overview
- Balanced subset cells: **15074**

Disease binary label counts in balanced subset:
|   y_ibd |   n_cells |
|--------:|----------:|
|       1 |      9895 |
|       0 |      5179 |

Top cell types in balanced subset:
| cell_type_label                         |   n_cells |
|:----------------------------------------|----------:|
| T cell of anorectum                     |      3600 |
| plasma cell                             |      3524 |
| colon epithelial cell                   |      3053 |
| myeloid cell                            |      2549 |
| stromal cell of lamina propria of colon |      2348 |

## Raw metadata label count file exists
The file `data/metadata_audit/ibd_candidate_label_counts.csv` contains the detailed raw counts for disease labels, cell types, donors, assay, tissue, and dataset fields.


# 2. Disease prediction
## Overall prediction summary
| embedding   | split_type       | cell_type_subset                        |   auroc_mean |   auroc_std |   auprc_mean |   balacc_mean |   n_runs |
|:------------|:-----------------|:----------------------------------------|-------------:|------------:|-------------:|--------------:|---------:|
| hvg         | celltype_matched | stromal cell of lamina propria of colon |     0.98656  |  0.00444295 |     0.991602 |      0.941641 |        5 |
| hvg         | celltype_matched | colon epithelial cell                   |     0.967816 |  0.00480539 |     0.980995 |      0.900083 |        5 |
| hvg         | celltype_matched | myeloid cell                            |     0.958977 |  0.00473206 |     0.985415 |      0.876326 |        5 |
| hvg         | celltype_matched | plasma cell                             |     0.941778 |  0.00974799 |     0.970252 |      0.876596 |        5 |
| hvg         | celltype_matched | T cell of anorectum                     |     0.919007 |  0.0119764  |     0.958257 |      0.845    |        5 |
| pca         | celltype_matched | stromal cell of lamina propria of colon |     0.975037 |  0.00433984 |     0.985249 |      0.922523 |        5 |
| pca         | celltype_matched | colon epithelial cell                   |     0.968796 |  0.00361365 |     0.981439 |      0.904865 |        5 |
| pca         | celltype_matched | myeloid cell                            |     0.942364 |  0.00464232 |     0.97979  |      0.869778 |        5 |
| pca         | celltype_matched | T cell of anorectum                     |     0.922165 |  0.0116096  |     0.959359 |      0.847708 |        5 |
| pca         | celltype_matched | plasma cell                             |     0.919703 |  0.00848366 |     0.958888 |      0.837234 |        5 |
| scvi        | celltype_matched | stromal cell of lamina propria of colon |     0.97567  |  0.0063531  |     0.984789 |      0.918909 |        5 |
| scvi        | celltype_matched | colon epithelial cell                   |     0.967022 |  0.00458879 |     0.979822 |      0.902316 |        5 |
| scvi        | celltype_matched | myeloid cell                            |     0.938453 |  0.00327083 |     0.977935 |      0.860741 |        5 |
| scvi        | celltype_matched | plasma cell                             |     0.925537 |  0.0136541  |     0.960427 |      0.849574 |        5 |
| scvi        | celltype_matched | T cell of anorectum                     |     0.89726  |  0.011333   |     0.944537 |      0.8125   |        5 |
| hvg         | donor_held_out   | all                                     |     0.858169 |  0.0687901  |     0.851869 |      0.771577 |        5 |
| pca         | donor_held_out   | all                                     |     0.857244 |  0.0830133  |     0.864821 |      0.765965 |        5 |
| scvi        | donor_held_out   | all                                     |     0.916363 |  0.0417641  |     0.915171 |      0.834044 |        5 |
| hvg         | random           | all                                     |     0.934097 |  0.00205147 |     0.964351 |      0.859734 |        5 |
| pca         | random           | all                                     |     0.908064 |  0.00532406 |     0.949867 |      0.827582 |        5 |
| scvi        | random           | all                                     |     0.915985 |  0.00418456 |     0.953821 |      0.836916 |        5 |

## Main all-cell split comparison
| embedding   | split_type     |   auroc_mean |   auroc_std |   n_runs |
|:------------|:---------------|-------------:|------------:|---------:|
| hvg         | donor_held_out |     0.858169 |  0.0687901  |        5 |
| hvg         | random         |     0.934097 |  0.00205147 |        5 |
| pca         | donor_held_out |     0.857244 |  0.0830133  |        5 |
| pca         | random         |     0.908064 |  0.00532406 |        5 |
| scvi        | donor_held_out |     0.916363 |  0.0417641  |        5 |
| scvi        | random         |     0.915985 |  0.00418456 |        5 |

## Best cell-type-matched disease prediction results
| embedding   | cell_type_subset                        |   auroc_mean |   auroc_std |   n_runs |
|:------------|:----------------------------------------|-------------:|------------:|---------:|
| hvg         | stromal cell of lamina propria of colon |     0.98656  |  0.00444295 |        5 |
| scvi        | stromal cell of lamina propria of colon |     0.97567  |  0.0063531  |        5 |
| pca         | stromal cell of lamina propria of colon |     0.975037 |  0.00433984 |        5 |
| pca         | colon epithelial cell                   |     0.968796 |  0.00361365 |        5 |
| hvg         | colon epithelial cell                   |     0.967816 |  0.00480539 |        5 |
| scvi        | colon epithelial cell                   |     0.967022 |  0.00458879 |        5 |
| hvg         | myeloid cell                            |     0.958977 |  0.00473206 |        5 |
| pca         | myeloid cell                            |     0.942364 |  0.00464232 |        5 |
| hvg         | plasma cell                             |     0.941778 |  0.00974799 |        5 |
| scvi        | myeloid cell                            |     0.938453 |  0.00327083 |        5 |
| scvi        | plasma cell                             |     0.925537 |  0.0136541  |        5 |
| pca         | T cell of anorectum                     |     0.922165 |  0.0116096  |        5 |
| pca         | plasma cell                             |     0.919703 |  0.00848366 |        5 |
| hvg         | T cell of anorectum                     |     0.919007 |  0.0119764  |        5 |
| scvi        | T cell of anorectum                     |     0.89726  |  0.011333   |        5 |


# 3. Leakage sensitivity
| embedding   |   donor_held_out |   random |   donor_leakage_sensitivity |
|:------------|-----------------:|---------:|----------------------------:|
| hvg         |         0.858169 | 0.934097 |                 0.0759284   |
| pca         |         0.857244 | 0.908064 |                 0.0508199   |
| scvi        |         0.916363 | 0.915985 |                -0.000378049 |


# 4. Confounder prediction
Metric used: **macro_f1**

| embedding   | task      |   metric_mean |   metric_std |   n_runs |
|:------------|:----------|--------------:|-------------:|---------:|
| hvg         | cell_type |      0.994985 |  0.000989759 |        5 |
| hvg         | disease   |      0.850363 |  0.0054555   |        5 |
| hvg         | donor     |      0.634271 |  0.00415067  |        5 |
| pca         | cell_type |      0.99223  |  0.00106728  |        5 |
| pca         | disease   |      0.809994 |  0.00767661  |        5 |
| pca         | donor     |      0.504563 |  0.0122731   |        5 |
| scvi        | cell_type |      0.993715 |  0.000691474 |        5 |
| scvi        | disease   |      0.822208 |  0.00365284  |        5 |
| scvi        | donor     |      0.445268 |  0.00562382  |        5 |

## Confounder burden file
| embedding   |   confounder_burden_mean_macro_f1 |
|:------------|----------------------------------:|
| hvg         |                          0.814628 |
| pca         |                          0.748396 |
| scvi        |                          0.719492 |


# 5. Negative controls


## Shuffled disease labels, PCA
| index   |      auroc |      auprc |   balanced_accuracy |        f1 |
|:--------|-----------:|-----------:|--------------------:|----------:|
| mean    | 0.489728   | 0.650728   |          0.489638   | 0.571543  |
| std     | 0.00454049 | 0.00729539 |          0.00688819 | 0.0249775 |
| min     | 0.485685   | 0.642966   |          0.4781     | 0.529758  |
| max     | 0.494803   | 0.659383   |          0.496398   | 0.597275  |


## Shuffled labels within cell type, PCA
| index   |      auroc |      auprc |   balanced_accuracy |        f1 |
|:--------|-----------:|-----------:|--------------------:|----------:|
| mean    | 0.535028   | 0.685804   |          0.526869   | 0.603749  |
| std     | 0.00823946 | 0.00732218 |          0.00843118 | 0.0155058 |
| min     | 0.52777    | 0.676695   |          0.52085    | 0.587732  |
| max     | 0.546659   | 0.694431   |          0.540259   | 0.621807  |


## Cell-type-proportion-only classifier
| index   |    auroc |    auprc |   balanced_accuracy |        f1 |
|:--------|---------:|---------:|--------------------:|----------:|
| mean    | 0.825    | 0.9275   |            0.75     | 0.769524  |
| std     | 0.167705 | 0.080558 |            0.153093 | 0.0967499 |
| min     | 0.625    | 0.804167 |            0.5      | 0.666667  |
| max     | 1        | 1        |            0.875    | 0.857143  |


## Dataset-ID-only classifier
_File missing or empty._


# 6. Pseudobulk validation
- Pseudobulk samples: **90**

Top pseudobulk cell types:
| cell_type_label                         |   n_pseudobulk_samples |
|:----------------------------------------|-----------------------:|
| T cell of anorectum                     |                     18 |
| colon epithelial cell                   |                     18 |
| myeloid cell                            |                     18 |
| plasma cell                             |                     18 |
| stromal cell of lamina propria of colon |                     18 |

Pseudobulk prediction results:
|   seed | split_type                |   n_train |   n_test |   auroc |   auprc |   balanced_accuracy |       f1 |
|-------:|:--------------------------|----------:|---------:|--------:|--------:|--------------------:|---------:|
|      0 | donor_held_out_pseudobulk |        70 |       20 |       1 |       1 |            0.933333 | 0.833333 |
|      1 | donor_held_out_pseudobulk |        70 |       20 |       1 |       1 |            0.9      | 0.967742 |
|      2 | donor_held_out_pseudobulk |        70 |       20 |       1 |       1 |            0.966667 | 0.965517 |
|      3 | donor_held_out_pseudobulk |        70 |       20 |       1 |       1 |            1        | 1        |
|      4 | donor_held_out_pseudobulk |        70 |       20 |       1 |       1 |            0.9      | 0.909091 |


# 7. Pathway validation
## Pathway gene presence
| pathway                    |   n_requested |   n_present | present_genes                                                        |
|:---------------------------|--------------:|------------:|:---------------------------------------------------------------------|
| tnf_nfkb_score             |            10 |          10 | TNF;NFKBIA;NFKB1;RELA;CXCL8;IL1B;CXCL2;CXCL3;CCL2;ICAM1              |
| interferon_score           |            13 |          13 | ISG15;IFIT1;IFIT2;IFIT3;MX1;MX2;OAS1;OAS2;OAS3;IFI6;IFI27;IRF7;STAT1 |
| antigen_presentation_score |             9 |           9 | HLA-DRA;HLA-DRB1;HLA-DPA1;HLA-DPB1;HLA-A;HLA-B;HLA-C;CD74;B2M        |
| epithelial_stress_score    |            10 |          10 | REG1A;REG1B;DUOX2;LCN2;MUC1;MUC2;KRT8;KRT18;CLDN1;CLDN2              |
| myeloid_inflammation_score |            10 |          10 | LYZ;S100A8;S100A9;FCGR3A;FCGR2A;IL1B;CXCL8;LST1;TYROBP;CTSS          |

## Pathway correlations
| pathway                    | target            |     n |   spearman_r |      p_value |   note |     abs_r |
|:---------------------------|:------------------|------:|-------------:|-------------:|-------:|----------:|
| antigen_presentation_score | pca_oof_ibd_score | 10127 |    0.224765  | 3.71713e-116 |    nan | 0.224765  |
| myeloid_inflammation_score | pca_oof_ibd_score | 10127 |    0.213832  | 4.64815e-105 |    nan | 0.213832  |
| antigen_presentation_score | y_ibd             | 15074 |    0.165753  | 2.60259e-93  |    nan | 0.165753  |
| tnf_nfkb_score             | pca_oof_ibd_score | 10127 |    0.129842  | 2.51792e-39  |    nan | 0.129842  |
| interferon_score           | pca_oof_ibd_score | 10127 |    0.115058  | 3.424e-31    |    nan | 0.115058  |
| myeloid_inflammation_score | y_ibd             | 15074 |    0.104601  | 6.08514e-38  |    nan | 0.104601  |
| interferon_score           | y_ibd             | 15074 |    0.0488321 | 1.99143e-09  |    nan | 0.0488321 |
| epithelial_stress_score    | pca_oof_ibd_score | 10127 |    0.0234022 | 0.0185193    |    nan | 0.0234022 |
| tnf_nfkb_score             | y_ibd             | 15074 |    0.0182067 | 0.0253942    |    nan | 0.0182067 |
| epithelial_stress_score    | y_ibd             | 15074 |   -0.0105999 | 0.19314      |    nan | 0.0105999 |


# 8. Automatic interpretation and proposed next steps
## Strengths found
- Best cell-type-matched signal is **stromal cell of lamina propria of colon** with AUROC 0.987. This cell type should be biologically interpreted first.
- For **hvg**, donor-held-out AUROC is fairly strong (0.858), suggesting some disease signal generalizes across donors.
- For **pca**, donor-held-out AUROC is fairly strong (0.857), suggesting some disease signal generalizes across donors.
- For **scvi**, donor-held-out AUROC is fairly strong (0.916), suggesting some disease signal generalizes across donors.
- Negative control **Shuffled disease labels, PCA** is near chance AUROC 0.490, which supports pipeline validity.
- Negative control **Shuffled labels within cell type, PCA** is near chance AUROC 0.535, which supports pipeline validity.
- Pseudobulk AUROC is strong (1.000), suggesting signal survives donor/cell-type aggregation.

## Warnings / interpretation cautions
- Cell-type proportions alone predict IBD with AUROC 0.825. Composition is a major confounder.
- The strongest confounder prediction is **cell_type** for **hvg** with score 0.995. This means embeddings strongly encode non-disease structure.

## Recommended next steps
1. Compare full model performance against the cell-type-proportion-only baseline in the main results.
2. Do pathway validation specifically within **stromal cell of lamina propria of colon** cells, not only across all cells.
3. High-impact next step: add a second IBD/gut dataset so you can run true dataset-held-out cross-atlas generalization.
4. Keep the confounder heatmap as a main figure. It is central to the benchmark argument.
5. Pathway correlations are weak. Try cell-type-specific pathway validation, especially myeloid and epithelial compartments.
6. Since scVI embeddings exist, compare HVG vs PCA vs scVI and decide whether foundation model embeddings are worth adding next.
