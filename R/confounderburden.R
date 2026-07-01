packages <- c("ggplot2")

for (pkg in packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg)
  }
}

library(ggplot2)

source_file <- "C:/Users/activ/OneDrive/Pictures/scDiseaseShift_IBD_starter_FULL_PROJECT/scDiseaseShift_IBD_starter/_archive_pre_6dataset/results_old/paper_tables/table_confounder_prediction.csv"
out_dir <- "C:/Users/activ/OneDrive/Documents/ML R2/ibd_manuscript_outputs/figures"
out_file <- file.path(out_dir, "confounder_predictability_heatmap_actual_data.png")

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

confounder_df <- read.csv(source_file, stringsAsFactors = FALSE)

confounder_df$embedding <- factor(
  confounder_df$embedding,
  levels = c("hvg", "pca", "scvi"),
  labels = c("HVG", "PCA", "scVI")
)

confounder_df$task <- factor(
  confounder_df$task,
  levels = c("cell_type", "disease", "donor"),
  labels = c("Cell type", "Disease", "Donor")
)

confounder_df$tile_label <- paste0(
  sprintf("%.3f", confounder_df$score_mean),
  "\n+/- ",
  sprintf("%.3f", confounder_df$score_sd)
)

p <- ggplot(confounder_df, aes(x = task, y = embedding, fill = score_mean)) +
  geom_tile(color = "white", linewidth = 1.2) +
  geom_text(
    aes(label = tile_label),
    size = 4.2,
    color = "black",
    lineheight = 0.9
  ) +
  scale_fill_gradientn(
    colors = c("#440154", "#31688e", "#35b779", "#fde725"),
    limits = c(0.40, 1.00),
    breaks = seq(0.4, 1.0, 0.1),
    name = "Macro F1"
  ) +
  labs(
    title = "",
    subtitle = "",
    x = NULL,
    y = NULL
  ) +
  coord_fixed(ratio = 1.15) +
  theme_minimal(base_size = 13) +
  theme(
    plot.title = element_text(face = "bold", size = 16, hjust = 0.5),
    plot.subtitle = element_text(size = 11, hjust = 0.5),
    axis.text.x = element_text(angle = 35, hjust = 1, size = 12),
    axis.text.y = element_text(size = 12),
    panel.grid = element_blank(),
    legend.title = element_text(face = "bold"),
    legend.position = "right",
    plot.margin = margin(t = 15, r = 15, b = 15, l = 15)
  )

print(p)

ggplot2::ggsave(
  filename = out_file,
  plot = p,
  width = 8.5,
  height = 5.5,
  dpi = 300
)

message("Saved: ", out_file)
