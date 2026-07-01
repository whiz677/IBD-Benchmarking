# =========================
# Donor-sample aggregation benchmark
# Unit: donor_label + sample_label
# Score: mean predicted IBD probability across cells
# =========================

library(ggplot2)
library(dplyr)
library(tidyr)

df <- tibble::tribble(
  ~Setting, ~Units, ~IBD_units, ~Control_units, ~Model, ~AUROC, ~AUPRC,
  
  "Martin all cells", 22, 11, 11, "PCA",        0.826, 0.874,
  "Martin all cells", 22, 11, 11, "scVI",       0.884, 0.908,
  "Martin all cells", 22, 11, 11, "scGPT",      0.893, 0.894,
  "Martin all cells", 22, 11, 11, "Geneformer", 0.579, 0.613,
  
  "Martin epithelial", 22, 11, 11, "PCA",        0.818, 0.875,
  "Martin epithelial", 22, 11, 11, "scVI",       0.909, 0.927,
  "Martin epithelial", 22, 11, 11, "scGPT",      0.860, 0.850,
  "Martin epithelial", 22, 11, 11, "Geneformer", 0.661, 0.683,
  
  "Oliver-no-Martin all cells", 130, 25, 105, "PCA",        0.744, 0.514,
  "Oliver-no-Martin all cells", 130, 25, 105, "scVI",       0.740, 0.523,
  "Oliver-no-Martin all cells", 130, 25, 105, "scGPT",      0.730, 0.428,
  "Oliver-no-Martin all cells", 130, 25, 105, "Geneformer", 0.318, 0.211,
  
  "Oliver-no-Martin epithelial", 130, 25, 105, "PCA",        0.733, 0.463,
  "Oliver-no-Martin epithelial", 130, 25, 105, "scVI",       0.768, 0.538,
  "Oliver-no-Martin epithelial", 130, 25, 105, "scGPT",      0.707, 0.382,
  "Oliver-no-Martin epithelial", 130, 25, 105, "Geneformer", 0.269, 0.154
)

plot_df <- df %>%
  mutate(
    Setting_label = paste0(
      Setting,
      "\n",
      Units, " donor-sample units; ",
      IBD_units, " IBD / ",
      Control_units, " control"
    ),
    Setting_label = factor(
      Setting_label,
      levels = rev(c(
        "Martin all cells\n22 donor-sample units; 11 IBD / 11 control",
        "Martin epithelial\n22 donor-sample units; 11 IBD / 11 control",
        "Oliver-no-Martin all cells\n130 donor-sample units; 25 IBD / 105 control",
        "Oliver-no-Martin epithelial\n130 donor-sample units; 25 IBD / 105 control"
      ))
    ),
    Model = factor(Model, levels = c("PCA", "scVI", "scGPT", "Geneformer"))
  ) %>%
  pivot_longer(
    cols = c(AUROC, AUPRC),
    names_to = "Metric",
    values_to = "Value"
  ) %>%
  mutate(
    Metric = factor(Metric, levels = c("AUROC", "AUPRC"))
  )

p <- ggplot(plot_df, aes(x = Value, y = Setting_label, color = Model)) +
  geom_vline(
    data = data.frame(Metric = c("AUROC", "AUPRC")),
    aes(xintercept = 0.5),
    linetype = "dashed",
    color = "grey55",
    linewidth = 0.5,
    inherit.aes = FALSE
  ) +
  geom_point(
    position = position_dodge(width = 0.62),
    size = 3.6
  ) +
  geom_text(
    aes(label = sprintf("%.3f", Value)),
    position = position_dodge(width = 0.62),
    hjust = -0.25,
    size = 3.1,
    show.legend = FALSE
  ) +
  facet_wrap(~ Metric, nrow = 1) +
  scale_color_manual(
    values = c(
      "PCA" = "#66C2A5",
      "scVI" = "#FC8D62",
      "scGPT" = "#8DA0CB",
      "Geneformer" = "#D783C2"
    )
  ) +
  scale_x_continuous(
    limits = c(0.10, 1.00),
    breaks = seq(0.1, 1.0, 0.1)
  ) +
  labs(
    title = "Donor-sample aggregation of locked external predictions",
    subtitle = "Scores are mean predicted IBD probability within each donor_label + sample_label unit",
    x = "Donor-sample performance",
    y = NULL,
    color = "Model"
  ) +
  guides(
    color = guide_legend(
      title.position = "top",
      title.hjust = 0.5,
      nrow = 1
    )
  ) +
  theme_classic(base_size = 14) +
  theme(
    plot.title = element_text(face = "bold", size = 18),
    plot.subtitle = element_text(size = 12, margin = margin(b = 12)),
    
    legend.position = "top",
    legend.title = element_text(face = "bold"),
    legend.text = element_text(size = 11),
    
    strip.background = element_rect(fill = "grey92", color = "grey60"),
    strip.text = element_text(face = "bold", size = 13),
    
    axis.text.y = element_text(size = 10),
    axis.text.x = element_text(size = 10),
    axis.title.x = element_text(face = "bold"),
    
    plot.margin = margin(18, 30, 18, 25)
  )

print(p)

ggsave(
  filename = "donor_sample_aggregation_auroc_auprc.png",
  plot = p,
  width = 12,
  height = 5.8,
  dpi = 300,
  bg = "white"
)