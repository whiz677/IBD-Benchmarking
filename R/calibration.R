# =========================
# Calibration analysis
# Separate clean bar plots for Brier score and ECE
# =========================

library(ggplot2)
library(dplyr)
library(tidyr)
library(patchwork)

df <- tibble::tribble(
  ~Setting, ~Model, ~Cells, ~Positive_rate, ~Brier, ~ECE,
  
  "Martin all cells",             "PCA",        5462,  0.650, 0.196, 0.087,
  "Martin all cells",             "scVI",       5462,  0.650, 0.204, 0.092,
  "Martin all cells",             "scGPT",      5462,  0.650, 0.242, 0.157,
  "Martin all cells",             "Geneformer", 5462,  0.650, 0.263, 0.161,
  
  "Martin epithelial",            "PCA",        5384,  0.646, 0.197, 0.109,
  "Martin epithelial",            "scVI",       5384,  0.646, 0.189, 0.075,
  "Martin epithelial",            "scGPT",      5384,  0.646, 0.259, 0.189,
  "Martin epithelial",            "Geneformer", 5384,  0.646, 0.273, 0.221,
  
  "Oliver-no-Martin all cells",   "PCA",        33302, 0.337, 0.256, 0.250,
  "Oliver-no-Martin all cells",   "scVI",       33302, 0.337, 0.270, 0.251,
  "Oliver-no-Martin all cells",   "scGPT",      33302, 0.337, 0.320, 0.305,
  "Oliver-no-Martin all cells",   "Geneformer", 33302, 0.337, 0.292, 0.240,
  
  "Oliver-no-Martin epithelial",  "PCA",        33279, 0.336, 0.303, 0.284,
  "Oliver-no-Martin epithelial",  "scVI",       33279, 0.336, 0.274, 0.269,
  "Oliver-no-Martin epithelial",  "scGPT",      33279, 0.336, 0.383, 0.381,
  "Oliver-no-Martin epithelial",  "Geneformer", 33279, 0.336, 0.288, 0.218
)

df <- df %>%
  mutate(
    Setting = factor(
      Setting,
      levels = c(
        "Martin all cells",
        "Martin epithelial",
        "Oliver-no-Martin all cells",
        "Oliver-no-Martin epithelial"
      )
    ),
    Model = factor(Model, levels = c("PCA", "scVI", "scGPT", "Geneformer"))
  )

model_colors <- c(
  "PCA" = "#66C2A5",
  "scVI" = "#FC8D62",
  "scGPT" = "#8DA0CB",
  "Geneformer" = "#D783C2"
)

make_plot <- function(metric_name, y_label, y_max) {
  ggplot(df, aes(x = Setting, y = .data[[metric_name]], fill = Model)) +
    geom_col(
      position = position_dodge(width = 0.78),
      width = 0.68,
      color = "black",
      linewidth = 0.25
    ) +
    geom_text(
      aes(label = sprintf("%.3f", .data[[metric_name]])),
      position = position_dodge(width = 0.78),
      vjust = -0.35,
      size = 2.8
    ) +
    scale_fill_manual(values = model_colors) +
    scale_y_continuous(
      limits = c(0, y_max),
      breaks = seq(0, y_max, 0.1),
      expand = expansion(mult = c(0, 0.06))
    ) +
    labs(
      title = y_label,
      x = NULL,
      y = y_label,
      fill = "Model"
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
      plot.title = element_text(face = "bold", size = 14, hjust = 0.5),
      legend.position = "top",
      legend.title = element_text(face = "bold", size = 11),
      legend.text = element_text(size = 10),
      axis.text.x = element_text(angle = 30, hjust = 1, size = 9),
      axis.text.y = element_text(size = 9),
      axis.title.y = element_text(face = "bold", size = 11),
      plot.margin = margin(10, 15, 10, 15)
    )
}

p_brier <- make_plot("Brier", "Brier score", 0.45)
p_ece <- make_plot("ECE", "ECE", 0.45)

final_plot <- p_brier / p_ece +
  plot_layout(guides = "collect") &
  theme(
    legend.position = "top",
    legend.title = element_text(face = "bold")
  )

print(final_plot)

ggsave(
  filename = "locked_external_calibration_brier_ece_separate_bars.png",
  plot = final_plot,
  width = 11,
  height = 8,
  dpi = 300,
  bg = "white"
)