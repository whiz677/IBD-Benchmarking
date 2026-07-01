install.packages(c("ggplot2", "patchwork", "RColorBrewer"))

library(ggplot2)
library(patchwork)
library(RColorBrewer)

benchmark_df <- data.frame(
  setting = c(
    "Martin all cells", "Martin all cells", "Martin all cells", "Martin all cells",
    "Oliver-no-Martin all cells", "Oliver-no-Martin all cells", "Oliver-no-Martin all cells", "Oliver-no-Martin all cells",
    "Martin epithelial", "Martin epithelial", "Martin epithelial", "Martin epithelial",
    "Oliver-no-Martin epithelial", "Oliver-no-Martin epithelial", "Oliver-no-Martin epithelial", "Oliver-no-Martin epithelial"
  ),
  model = c(
    "PCA", "scVI", "scGPT", "Geneformer",
    "PCA", "scVI", "scGPT", "Geneformer",
    "PCA", "scVI", "scGPT", "Geneformer",
    "PCA", "scVI", "scGPT", "Geneformer"
  ),
  AUROC = c(
    0.766, 0.743, 0.701, 0.558,
    0.715, 0.673, 0.662, 0.408,
    0.778, 0.766, 0.673, 0.643,
    0.683, 0.723, 0.631, 0.484
  ),
  AUPRC = c(
    0.862, 0.838, 0.823, 0.719,
    0.661, 0.590, 0.569, 0.322,
    0.865, 0.851, 0.796, 0.762,
    0.590, 0.673, 0.537, 0.361
  )
)

benchmark_df$setting <- factor(
  benchmark_df$setting,
  levels = c(
    "Martin all cells",
    "Oliver-no-Martin all cells",
    "Martin epithelial",
    "Oliver-no-Martin epithelial"
  )
)

benchmark_df$model <- factor(
  benchmark_df$model,
  levels = c(
    "PCA",
    "scVI",
    "scGPT",
    "Geneformer"
  )
)

p_auroc <- ggplot(benchmark_df, aes(x = setting, y = AUROC, fill = model)) +
  geom_hline(yintercept = 0.5, linetype = "dashed", color = "gray55") +
  geom_col(
    position = position_dodge(width = 0.8),
    width = 0.7,
    color = "black",
    linewidth = 0.2
  ) +
  geom_text(
    aes(label = sprintf("%.3f", AUROC)),
    position = position_dodge(width = 0.8),
    vjust = -0.4,
    size = 2.5
  ) +
  scale_fill_brewer(palette = "Set2") +
  guides(
    fill = guide_legend(
      title.position = "top",
      title.hjust = 0.5
    )
  ) +
  scale_y_continuous(
    limits = c(0, 1),
    breaks = seq(0, 1, 0.2),
    expand = expansion(mult = c(0, 0.08))
  ) +
  labs(
    title = "",
    x = NULL,
    y = "AUROC",
    fill = "Model"
  ) +
  theme_classic(base_size = 12) +
  theme(
    axis.text.x = element_text(angle = 35, hjust = 1),
    legend.position = "top",
    legend.title = element_text(face = "bold"),
    legend.box = "vertical",
    plot.margin = margin(t = 15, r = 10, b = 15, l = 10)
  )

p_auprc <- ggplot(benchmark_df, aes(x = setting, y = AUPRC, fill = model)) +
  geom_col(
    position = position_dodge(width = 0.8),
    width = 0.7,
    color = "black",
    linewidth = 0.2
  ) +
  geom_text(
    aes(label = sprintf("%.3f", AUPRC)),
    position = position_dodge(width = 0.8),
    vjust = -0.4,
    size = 2.5
  ) +
  scale_fill_brewer(palette = "Set2") +
  guides(
    fill = guide_legend(
      title.position = "top",
      title.hjust = 0.5
    )
  ) +
  scale_y_continuous(
    limits = c(0, 1),
    breaks = seq(0, 1, 0.2),
    expand = expansion(mult = c(0, 0.08))
  ) +
  labs(
    title = "",
    x = NULL,
    y = "AUPRC",
    fill = "Model"
  ) +
  theme_classic(base_size = 12) +
  theme(
    axis.text.x = element_text(angle = 35, hjust = 1),
    legend.position = "top",
    legend.title = element_text(face = "bold"),
    legend.box = "vertical",
    plot.margin = margin(t = 15, r = 10, b = 15, l = 10)
  )

final_plot <- p_auroc / p_auprc +
  plot_layout(guides = "collect") &
  theme(
    legend.position = "top",
    legend.title = element_text(face = "bold"),
    legend.box = "vertical"
  )

print(final_plot)

ggplot2::ggsave(
  filename = "locked_external_benchmark_two_panel.png",
  plot = final_plot,
  width = 11,
  height = 8,
  dpi = 300
)
final_plot <- p_auroc / p_auprc

print(final_plot)

ggplot2::ggsave(
  filename = "locked_external_benchmark_two_panel.png",
  plot = final_plot,
  width = 11,
  height = 9,
  dpi = 300
)