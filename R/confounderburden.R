# =========================
# Current confounder burden figure
# Uses the non-archive leakage summary, not the old task-level heatmap
# =========================

packages <- c("ggplot2")

for (pkg in packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg)
  }
}

library(ggplot2)

source_file <- "C:/Users/activ/OneDrive/Pictures/scDiseaseShift_IBD_starter_FULL_PROJECT/scDiseaseShift_IBD_starter/results/leakage_scores/ibd_confounder_burden.csv"
out_dir <- "C:/Users/activ/OneDrive/Documents/ML R2/ibd_manuscript_outputs/figures"
out_file <- file.path(out_dir, "confounder_burden_current_mean_macro_f1.png")

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

df <- read.csv(source_file, stringsAsFactors = FALSE)

df$embedding <- factor(
  df$embedding,
  levels = c("hvg", "pca", "scvi"),
  labels = c("HVG", "PCA", "scVI")
)

df$label <- sprintf("%.3f", df$confounder_burden_mean_macro_f1)

p <- ggplot(
  df,
  aes(x = embedding, y = confounder_burden_mean_macro_f1, fill = embedding)
) +
  geom_col(width = 0.62, color = "black", linewidth = 0.25) +
  geom_text(aes(label = label), vjust = -0.45, size = 4.1) +
  scale_fill_manual(
    values = c("HVG" = "#66C2A5", "PCA" = "#8DA0CB", "scVI" = "#FC8D62"),
    guide = "none"
  ) +
  scale_y_continuous(
    limits = c(0, 1.0),
    breaks = seq(0, 1.0, 0.2),
    expand = expansion(mult = c(0, 0.03))
  ) +
  labs(
    title = "Confounder burden across early representations",
    subtitle = "Mean macro-F1 from current leakage-score summary",
    x = NULL,
    y = "Mean macro-F1"
  ) +
  theme_classic(base_size = 13) +
  theme(
    plot.title = element_text(face = "bold", size = 15, hjust = 0.5),
    plot.subtitle = element_text(size = 11, hjust = 0.5),
    axis.title.y = element_text(face = "bold"),
    axis.text = element_text(size = 11),
    plot.margin = margin(t = 15, r = 20, b = 15, l = 20)
  )

print(p)

ggplot2::ggsave(
  filename = out_file,
  plot = p,
  width = 6.2,
  height = 4.5,
  dpi = 300,
  bg = "white"
)

message("Saved: ", out_file)
