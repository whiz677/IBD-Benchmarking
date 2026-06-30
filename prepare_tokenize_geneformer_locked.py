train:
  - ./data/final_benchmark_matrices/train_dev_all_cells_common.h5ad

test:
  - ./data/final_benchmark_matrices/locked_test_martin_all_cells_common.h5ad
  - ./data/final_benchmark_matrices/locked_test_oliver_no_martin_all_cells_common.h5ad

label_key: y_ibd
batch_key: dataset_eval
celltype_key: cell_type_label