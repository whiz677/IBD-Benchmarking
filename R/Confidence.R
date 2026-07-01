packages <- c("ggplot2", "patchwork", "RColorBrewer", "cowplot")

for (pkg in packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg)
  }
}

library(ggplot2)
library(patchwork)
library(RColorBrewer)
library(cowplot)

auroc_ci_df <- data.frame(
  setting = c(
    "Martin all-cell", "Martin all-cell", "Martin all-cell", "Martin all-cell",
    "Oliver-no-Martin all-cell", "Oliver-no-Martin all-cell", "Oliver-no-Martin all-cell", "Oliver-no-Martin all-cell",
    "Martin epithelial", "Martin epithelial", "Martin epithelial", "Martin epithelial",
    "Oliver-no-Martin epithelial", "Oliver-no-Martin epithelial", "Oliver-no-Martin epithelial", "Oliver-no-Martin epithelial"
  ),
  model = c(
    "PCA", "scVI", "scGPT", "Geneformer",
    "PCA", "scVI", "scGPT", "Geneformer",
    "PCA", "scVI", "scGPT", "Geneformer",
    "PCA", "scVI", "scGPT", "Geneformer"
  ),
  auroc = c(
    0.766, 0.743, 0.701, 0.558,
    0.715, 0.673, 0.662, 0.408,
    0.778, 0.766, 0.673, 0.643,
    0.683, 0.723, 0.631, 0.484
  ),
  ci_low = c(
    0.753, 0.729, 0.687, 0.542,
    0.708, 0.666, 0.656, 0.401,
    0.766, 0.753, 0.658, 0.628,
    0.677, 0.717, 0.624, 0.477
  ),
  ci_high = c(
    0.778, 0.756, 0.715, 0.574,
    0.721, 0.679, 0.669, 0.415,
    0.791, 0.779, 0.687, 0.659,
    0.690, 0.729, 0.637, 0.491
  )
)

auroc_ci_df$model <- factor(
  auroc_ci_df$model,
  levels = c("PCA", "scVI", "scGPT", "Geneformer")
)

auroc_ci_df$label <- paste0(
  sprintf("%.3f", auroc_ci_df$auroc),
  " (",
  sprintf("%.3f", auroc_ci_df$ci_low),
  "-",
  sprintf("%.3f", auroc_ci_df$ci_high),
  ")"
)

auroc_ci_df$label_x <- auroc_ci_df$ci_high + 0.020

model_colors <- c(
  "PCA" = "#CC79A7",
  "scVI" = "#8DA0CB",
  "scGPT" = "#FC8D62",
  "Geneformer" = "#66C2A5"
)

legend_source <- ggplot(auroc_ci_df, aes(x = auroc, y = model, color = model)) +
  geom_point(size = 3.2) +
  scale_color_manual(values = model_colors) +
  labs(color = "Model") +
  guides(
    color = guide_legend(
      title.position = "top",
      title.hjust = 0.5,
      nrow = 1
    )
  ) +
  theme_void(base_size = 12) +
  theme(
    legend.position = "top",
    legend.title = element_text(face = "bold", size = 12),
    legend.text = element_text(size = 11),
    legend.box = "vertical"
  )

legend_only <- cowplot::get_legend(legend_source)

make_setting_plot <- function(setting_name) {
  setting_df <- auroc_ci_df[auroc_ci_df$setting == setting_name, ]
  
  ggplot(setting_df, aes(x = auroc, y = model, color = model)) +
    geom_vline(
      xintercept = 0.5,
      linetype = "dashed",
      color = "gray60",
      linewidth = 0.5
    ) +
    geom_errorbarh(
      aes(xmin = ci_low, xmax = ci_high),
      height = 0.16,
      linewidth = 0.9
    ) +
    geom_point(size = 3.2) +
    geom_text(
      aes(x = label_x, label = label),
      hjust = 0,
      size = 3.1,
      color = "black",
      family = "Arial"
    ) +
    scale_color_manual(values = model_colors) +
    scale_x_continuous(
      limits = c(0.35, 1.02),
      breaks = seq(0.4, 1.0, 0.1),
      expand = expansion(mult = c(0, 0.03))
    ) +
    labs(
      title = setting_name,
      x = "AUROC",
      y = NULL
    ) +
    theme_classic(base_size = 12) +
    theme(
      plot.title = element_text(face = "bold", size = 13, hjust = 0.5),
      panel.border = element_rect(color = "gray40", fill = NA, linewidth = 0.6),
      legend.position = "none",
      axis.text.y = element_text(size = 11),
      axis.text.x = element_text(size = 10),
      axis.title.x = element_text(size = 12),
      plot.margin = margin(t = 14, r = 45, b = 14, l = 8)
    )
}

p1 <- make_setting_plot("Martin all-cell")
p2 <- make_setting_plot("Oliver-no-Martin all-cell")
p3 <- make_setting_plot("Martin epithelial")
p4 <- make_setting_plot("Oliver-no-Martin epithelial")

final_plot <- wrap_elements(legend_only) / p1 / p2 / p3 / p4 +
  plot_layout(
    heights = c(0.28, 1.15, 1.15, 1.15, 1.15)
  )

print(final_plot)

ggplot2::ggsave(
  filename = "locked_external_auroc_delong_ci_four_panel.png",
  plot = final_plot,
  width = 11,
  height = 13.5,
  dpi = 300
)