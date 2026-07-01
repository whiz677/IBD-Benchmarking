packages <- c("ggplot2", "scales")

for (pkg in packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg)
  }
}

library(ggplot2)
library(scales)

zip_file <- "C:/Users/activ/OneDrive/Pictures/scDiseaseShift_IBD_starter_FULL_PROJECT/final_locked_external_finetune_results.zip"

finetune_inner_file <- "finetune_models/outputs/locked_external_summary/locked_external_model_comparison_with_ci.csv"

baseline_file <- "C:/Users/activ/OneDrive/Documents/ML R2/ibd_external_stats/locked_external_model_metrics_donor_bootstrap.csv"

out_dir <- "C:/Users/activ/OneDrive/Documents/ML R2/ibd_manuscript_outputs/figures"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

temp_dir <- tempdir()
unzip(zip_file, files = finetune_inner_file, exdir = temp_dir, overwrite = TRUE)

finetune_file <- file.path(temp_dir, finetune_inner_file)

finetune_df <- read.csv(finetune_file, stringsAsFactors = FALSE)
baseline_df <- read.csv(baseline_file, stringsAsFactors = FALSE)

finetune_df <- finetune_df[
  finetune_df$subset %in% c("martin", "oliver_no_martin") &
    finetune_df$model %in% c("scVI", "scGPT", "Geneformer"),
]

finetune_df$test_dataset <- finetune_df$subset
finetune_df$condition <- "Fine-tuned/adapted"

baseline_df <- baseline_df[
  baseline_df$group == "all_cells" &
    baseline_df$test_dataset %in% c("martin", "oliver_no_martin") &
    baseline_df$model %in% c("scVI_30", "scGPT_frozen", "Geneformer_frozen"),
]

baseline_df$model_clean <- baseline_df$model
baseline_df$model_clean[baseline_df$model == "scVI_30"] <- "scVI"
baseline_df$model_clean[baseline_df$model == "scGPT_frozen"] <- "scGPT"
baseline_df$model_clean[baseline_df$model == "Geneformer_frozen"] <- "Geneformer"

baseline_plot <- data.frame(
  model = baseline_df$model_clean,
  test_dataset = baseline_df$test_dataset,
  condition = "Frozen/baseline",
  auroc = baseline_df$auroc
)

finetune_plot <- data.frame(
  model = finetune_df$model,
  test_dataset = finetune_df$test_dataset,
  condition = "Fine-tuned/adapted",
  auroc = finetune_df$auroc
)

plot_df <- rbind(baseline_plot, finetune_plot)

plot_df$model <- factor(
  plot_df$model,
  levels = c("scVI", "scGPT", "Geneformer")
)

plot_df$test_clean <- plot_df$test_dataset
plot_df$test_clean[plot_df$test_dataset == "martin"] <- "Martin"
plot_df$test_clean[plot_df$test_dataset == "oliver_no_martin"] <- "Oliver-no-Martin"

plot_df$test_clean <- factor(
  plot_df$test_clean,
  levels = c("Martin", "Oliver-no-Martin")
)

plot_df$condition <- factor(
  plot_df$condition,
  levels = c("Frozen/baseline", "Fine-tuned/adapted")
)

condition_colors <- c(
  "Frozen/baseline" = "#9E9E9E",
  "Fine-tuned/adapted" = "#2C7FB8"
)

p <- ggplot(plot_df, aes(x = model, y = auroc, fill = condition)) +
  geom_col(
    position = position_dodge(width = 0.75),
    width = 0.65,
    color = "black",
    linewidth = 0.25
  ) +
  geom_text(
    aes(label = sprintf("%.3f", auroc)),
    position = position_dodge(width = 0.75),
    vjust = -0.35,
    size = 3.5
  ) +
  facet_wrap(~test_clean, nrow = 1) +
  scale_fill_manual(values = condition_colors, name = "Model state") +
  scale_y_continuous(
    limits = c(0, 0.85),
    breaks = seq(0, 0.8, 0.1),
    labels = number_format(accuracy = 0.1),
    expand = expansion(mult = c(0, 0.08))
  ) +
  labs(
    title = "",
    subtitle = "all-cell",
    x = NULL,
    y = "AUROC"
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
    plot.title = element_text(face = "bold", size = 16, hjust = 0.5),
    plot.subtitle = element_text(size = 11, hjust = 0.5),
    strip.text = element_text(face = "bold", size = 12),
    axis.text.x = element_text(size = 11),
    axis.title.y = element_text(face = "bold"),
    legend.position = "top",
    legend.title = element_text(face = "bold"),
    legend.box = "vertical",
    plot.margin = margin(t = 15, r = 15, b = 15, l = 15)
  )

print(p)

ggplot2::ggsave(
  filename = file.path(out_dir, "figure8_finetuning_sensitivity_auroc.png"),
  plot = p,
  width = 10,
  height = 6,
  dpi = 300
)