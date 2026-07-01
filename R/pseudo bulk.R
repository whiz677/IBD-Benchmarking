# =========================
# Pseudobulk donor-held-out benchmark
# No janitor needed
# =========================

library(ggplot2)
library(dplyr)
library(tidyr)

pseudobulk_file <- "C:/Users/activ/OneDrive/Pictures/scDiseaseShift_IBD_starter_FULL_PROJECT/scDiseaseShift_IBD_starter/results/pseudobulk/ibd_pseudobulk_hvg_logistic.csv"

df <- read.csv(pseudobulk_file, stringsAsFactors = FALSE)

# Clean column names without janitor
names(df) <- tolower(names(df))
names(df) <- gsub("[^a-z0-9]+", "_", names(df))
names(df) <- gsub("_$", "", names(df))

print(names(df))

# Find columns safely
seed_col <- intersect(c("seed", "random_seed"), names(df))[1]
auroc_col <- intersect(c("auroc", "roc_auc"), names(df))[1]
auprc_col <- intersect(c("auprc", "average_precision", "pr_auc"), names(df))[1]
balacc_col <- intersect(c("balanced_accuracy", "bal_acc"), names(df))[1]
f1_col <- intersect(c("f1", "f1_score"), names(df))[1]

plot_df <- data.frame(
  Seed = df[[seed_col]],
  AUROC = df[[auroc_col]],
  AUPRC = df[[auprc_col]],
  Balanced_accuracy = df[[balacc_col]],
  F1 = df[[f1_col]]
)

plot_df <- plot_df %>%
  mutate(
    Seed = factor(Seed)
  ) %>%
  pivot_longer(
    cols = c(AUROC, AUPRC, Balanced_accuracy, F1),
    names_to = "Metric",
    values_to = "Value"
  ) %>%
  mutate(
    Metric = recode(
      Metric,
      "Balanced_accuracy" = "Balanced accuracy"
    ),
    Metric = factor(
      Metric,
      levels = c("AUROC", "AUPRC", "Balanced accuracy", "F1")
    )
  )

p <- ggplot(plot_df, aes(x = Seed, y = Value, fill = Metric)) +
  geom_col(
    position = position_dodge(width = 0.78),
    width = 0.68,
    color = "black",
    linewidth = 0.25
  ) +
  geom_text(
    aes(label = sprintf("%.3f", Value)),
    position = position_dodge(width = 0.78),
    vjust = -0.35,
    size = 3.0
  ) +
  scale_fill_manual(
    values = c(
      "AUROC" = "#66C2A5",
      "AUPRC" = "#FC8D62",
      "Balanced accuracy" = "#8DA0CB",
      "F1" = "#D783C2"
    )
  ) +
  scale_y_continuous(
    limits = c(0, 1.08),
    breaks = seq(0, 1.0, 0.2),
    expand = expansion(mult = c(0, 0.02))
  ) +
  labs(
    title = "Pseudobulk donor-held-out benchmark",
    x = "Seed",
    y = "Performance",
    fill = "Metric"
  ) +
  guides(
    fill = guide_legend(
      title.position = "top",
      title.hjust = 0.5,
      nrow = 1
    )
  ) +
  theme_classic(base_size = 14) +
  theme(
    plot.title = element_text(face = "bold", size = 18),
    legend.position = "top",
    legend.title = element_text(face = "bold"),
    legend.text = element_text(size = 10),
    axis.title = element_text(face = "bold"),
    axis.text = element_text(size = 10),
    plot.margin = margin(15, 25, 15, 25)
  )

print(p)

ggsave(
  filename = "pseudobulk_donor_held_out_seed_performance.png",
  plot = p,
  width = 9.5,
  height = 5.2,
  dpi = 300,
  bg = "white"
)