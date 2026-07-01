packages <- c("ggplot2")

for (pkg in packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg)
  }
}

library(ggplot2)

source_file <- "C:/Users/activ/OneDrive/Pictures/scDiseaseShift_IBD_starter_FULL_PROJECT/scDiseaseShift_IBD_starter/_archive_pre_6dataset/results_old/scvi_pathway_validation/scvi_pathway_correlations.csv"
out_dir <- "C:/Users/activ/OneDrive/Documents/ML R2/ibd_manuscript_outputs/figures"
out_file <- file.path(out_dir, "scvi_pathway_correlation_heatmap_actual_data.png")

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

pathway_df <- read.csv(source_file, stringsAsFactors = FALSE)

# This follows the project figure logic: focus on interpretable, higher-performing
# cell compartments used for the original Figure 6.
pathway_df <- pathway_df[
  pathway_df$broad_cell_group %in% c("epithelial", "myeloid", "stromal_endothelial"),
]

pathway_df$condition <- paste0(
  pathway_df$broad_cell_group,
  "\n",
  pathway_df$train_dataset,
  " -> ",
  pathway_df$test_dataset
)

condition_levels <- c(
  "epithelial\ngarrido -> kong",
  "epithelial\nkong -> garrido",
  "myeloid\ngarrido -> kong",
  "myeloid\nkong -> garrido",
  "stromal_endothelial\ngarrido -> kong",
  "stromal_endothelial\nkong -> garrido"
)

pathway_levels <- c(
  "antigen_presentation",
  "epithelial_stress",
  "interferon",
  "myeloid_inflammation",
  "stromal_remodeling",
  "tnf_nfkb"
)

pathway_labels <- c(
  "antigen_presentation" = "Antigen\npresentation",
  "epithelial_stress" = "Epithelial\nstress",
  "interferon" = "Interferon",
  "myeloid_inflammation" = "Myeloid\ninflammation",
  "stromal_remodeling" = "Stromal\nremodeling",
  "tnf_nfkb" = "TNF/NF-kB"
)

condition_labels <- c(
  "epithelial\ngarrido -> kong" = "Epithelial\nGarrido -> Kong",
  "epithelial\nkong -> garrido" = "Epithelial\nKong -> Garrido",
  "myeloid\ngarrido -> kong" = "Myeloid\nGarrido -> Kong",
  "myeloid\nkong -> garrido" = "Myeloid\nKong -> Garrido",
  "stromal_endothelial\ngarrido -> kong" = "Stromal/endothelial\nGarrido -> Kong",
  "stromal_endothelial\nkong -> garrido" = "Stromal/endothelial\nKong -> Garrido"
)

pathway_df$condition <- factor(pathway_df$condition, levels = rev(condition_levels))
pathway_df$pathway <- factor(pathway_df$pathway, levels = pathway_levels)

pathway_df$label <- sprintf("%.2f", pathway_df$spearman_pathway_vs_scvi_score)

p <- ggplot(
  pathway_df,
  aes(x = pathway, y = condition, fill = spearman_pathway_vs_scvi_score)
) +
  geom_tile(color = "white", linewidth = 0.8) +
  geom_text(aes(label = label), size = 3.8, color = "black") +
  scale_fill_gradient2(
    low = "#440154",
    mid = "#2c7fb8",
    high = "#fde725",
    midpoint = 0,
    limits = c(-0.25, 0.65),
    breaks = seq(-0.2, 0.6, 0.2),
    name = "Spearman r"
  ) +
  scale_x_discrete(labels = pathway_labels) +
  scale_y_discrete(labels = condition_labels) +
  labs(
    title = "",
    subtitle = "",
    x = NULL,
    y = NULL
  ) +
  coord_fixed(ratio = 0.75) +
  theme_minimal(base_size = 13) +
  theme(
    plot.title = element_text(face = "bold", size = 16, hjust = 0.5),
    plot.subtitle = element_text(size = 11, hjust = 0.5),
    axis.text.x = element_text(angle = 40, hjust = 1, size = 11),
    axis.text.y = element_text(size = 11),
    panel.grid = element_blank(),
    legend.title = element_text(face = "bold"),
    legend.position = "right",
    plot.margin = margin(t = 15, r = 15, b = 15, l = 15)
  )

print(p)

ggplot2::ggsave(
  filename = out_file,
  plot = p,
  width = 11,
  height = 7,
  dpi = 300
)

message("Saved: ", out_file)
