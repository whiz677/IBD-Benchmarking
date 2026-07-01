packages <- c("ggplot2")

for (pkg in packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg)
  }
}

library(ggplot2)

source_file <- "C:/Users/activ/OneDrive/Pictures/scDiseaseShift_IBD_starter_FULL_PROJECT/scDiseaseShift_IBD_starter/_archive_pre_6dataset/results_old/final_publication_outputs/tables/Table6_shuffle_label_controls.csv"
out_dir <- "C:/Users/activ/OneDrive/Documents/ML R2/ibd_manuscript_outputs/figures"
out_file <- file.path(out_dir, "shuffle_label_control_real_minus_shuffle_actual_data.png")

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

shuffle_df <- read.csv(source_file, stringsAsFactors = FALSE)

plot_df <- shuffle_df[order(shuffle_df$real_minus_shuffle, decreasing = TRUE), ]
plot_df <- head(plot_df, 15)

clean_model <- function(x) {
  x <- gsub("_30$", "", x)
  x
}

clean_group <- function(x) {
  x <- gsub("_", "/", x)
  x
}

plot_df$label <- paste0(
  clean_model(plot_df$model),
  " | ",
  clean_group(plot_df$broad_cell_group),
  " | ",
  plot_df$train_dataset,
  " -> ",
  plot_df$test_dataset
)

plot_df$model_clean <- factor(
  clean_model(plot_df$model),
  levels = c("PCA", "scVI", "scGPT")
)

plot_df$label <- factor(plot_df$label, levels = rev(plot_df$label))

model_colors <- c(
  "PCA" = "#CC79A7",
  "scVI" = "#8DA0CB",
  "scGPT" = "#FC8D62"
)

p <- ggplot(
  plot_df,
  aes(x = real_minus_shuffle, y = label, fill = model_clean)
) +
  geom_col(width = 0.72, color = "black", linewidth = 0.2) +
  geom_text(
    aes(label = sprintf("%.3f", real_minus_shuffle)),
    hjust = -0.12,
    size = 3.6
  ) +
  geom_vline(xintercept = 0, color = "gray35", linewidth = 0.4) +
  scale_fill_manual(values = model_colors, name = "Model") +
  scale_x_continuous(
    limits = c(0, 0.38),
    breaks = seq(0, 0.35, 0.05),
    expand = expansion(mult = c(0, 0.02))
  ) +
  labs(
    title = "",
    subtitle = "",
    x = "Real AUROC - shuffled-label mean AUROC",
    y = NULL
  ) +
  guides(
    fill = guide_legend(
      title.position = "top",
      title.hjust = 0.5,
      nrow = 1
    )
  ) +
  theme_classic(base_size = 12) +
  theme(
    plot.title = element_text(face = "bold", size = 15, hjust = 0.5),
    plot.subtitle = element_text(size = 11, hjust = 0.5),
    axis.text.y = element_text(size = 10),
    axis.text.x = element_text(size = 10),
    legend.position = "top",
    legend.title = element_text(face = "bold"),
    legend.box = "vertical",
    plot.margin = margin(t = 15, r = 35, b = 15, l = 10)
  )

print(p)

ggplot2::ggsave(
  filename = out_file,
  plot = p,
  width = 10.5,
  height = 7.5,
  dpi = 300
)

message("Saved: ", out_file)
