# =========================
# Shuffle-label control plot: all tests
# Shows all 8 tests per model: PCA, scVI, and scGPT
# =========================

library(ggplot2)

source_file <- "C:/Users/activ/OneDrive/Pictures/scDiseaseShift_IBD_starter_FULL_PROJECT/scDiseaseShift_IBD_starter/_archive_pre_6dataset/results_old/final_publication_outputs/tables/Table6_shuffle_label_controls.csv"
out_dir <- "C:/Users/activ/OneDrive/Documents/ML R2/ibd_manuscript_outputs/figures"
out_file <- file.path(out_dir, "shuffle_label_control_all_24_tests.png")

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

shuffle_df <- read.csv(source_file, stringsAsFactors = FALSE)

clean_model <- function(x) {
  gsub("_30$", "", x)
}

clean_group <- function(x) {
  gsub("_", "/", x)
}

plot_df <- shuffle_df
plot_df$model_clean <- clean_model(plot_df$model)
plot_df$model_clean <- factor(plot_df$model_clean, levels = c("PCA", "scVI", "scGPT"))

plot_df$label <- paste0(
  clean_group(plot_df$broad_cell_group),
  " | ",
  plot_df$train_dataset,
  " -> ",
  plot_df$test_dataset
)

# Order labels within each model by real-minus-shuffle margin.
plot_df <- plot_df[order(plot_df$model_clean, plot_df$real_minus_shuffle), ]
plot_df$label_ordered <- factor(
  paste(plot_df$model_clean, plot_df$label, sep = " | "),
  levels = paste(plot_df$model_clean, plot_df$label, sep = " | ")
)

model_counts <- table(plot_df$model_clean)
print(model_counts)

model_colors <- c(
  "PCA" = "#CC79A7",
  "scVI" = "#8DA0CB",
  "scGPT" = "#FC8D62"
)

p <- ggplot(
  plot_df,
  aes(x = real_minus_shuffle, y = label_ordered, fill = model_clean)
) +
  geom_col(width = 0.72, color = "black", linewidth = 0.2) +
  geom_text(
    aes(label = sprintf("%.3f", real_minus_shuffle)),
    hjust = -0.10,
    size = 3.0
  ) +
  geom_vline(xintercept = 0, color = "gray35", linewidth = 0.4) +
  facet_grid(
    model_clean ~ .,
    scales = "free_y",
    space = "free_y"
  ) +
  scale_fill_manual(values = model_colors, guide = "none") +
  scale_x_continuous(
    limits = c(0, 0.38),
    breaks = seq(0, 0.35, 0.05),
    expand = expansion(mult = c(0, 0.02))
  ) +
  scale_y_discrete(
    labels = function(x) sub("^[^|]+ \\| ", "", x)
  ) +
  labs(
    title = "Shuffle-label control validation",
    subtitle = "All 24 tests shown: 8 PCA, 8 scVI, and 8 scGPT comparisons",
    x = "Real-minus-shuffle AUROC",
    y = NULL
  ) +
  theme_classic(base_size = 12) +
  theme(
    plot.title = element_text(face = "bold", size = 15, hjust = 0.5),
    plot.subtitle = element_text(size = 11, hjust = 0.5),
    strip.background = element_rect(fill = "gray90", color = "gray60"),
    strip.text.y = element_text(face = "bold", size = 11, angle = 0),
    axis.text.y = element_text(size = 8.5),
    axis.text.x = element_text(size = 10),
    axis.title.x = element_text(face = "bold"),
    plot.margin = margin(t = 15, r = 35, b = 15, l = 10)
  )

print(p)

ggplot2::ggsave(
  filename = out_file,
  plot = p,
  width = 10.5,
  height = 10.5,
  dpi = 300,
  bg = "white"
)

message("Saved: ", out_file)
