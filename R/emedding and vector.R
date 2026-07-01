# =========================
# Embedding disease-separation summary
# Fully labeled x-axes version
# =========================

library(ggplot2)
library(dplyr)
library(tidyr)
library(patchwork)

df <- tibble::tribble(
  ~Scope, ~Model, ~Cells, ~Dimensions, ~Max_abs_SMD, ~Mean_abs_SMD, ~Dims_SMD_0_2, ~Dims_SMD_0_5, ~Top10_signal_share,
  
  "All cells", "PCA",        72000, 50,   0.391, 0.124, 10,  0, 0.422,
  "All cells", "scVI",       72000, 30,   0.539, 0.188, 13,  1, 0.611,
  "All cells", "scGPT",      72000, 512,  0.573, 0.141, 138, 1, 0.063,
  "All cells", "Geneformer", 72000, 1152, 0.123, 0.029, 0,   0, 0.031,
  
  "Epithelial", "PCA",        39842, 50,   0.499, 0.119, 7,   0, 0.533,
  "Epithelial", "scVI",       39842, 30,   0.896, 0.256, 12,  5, 0.692,
  "Epithelial", "scGPT",      39842, 512,  0.668, 0.195, 218, 12, 0.058,
  "Epithelial", "Geneformer", 39842, 1152, 0.378, 0.111, 169, 0, 0.028
)

df <- df %>%
  mutate(
    Scope = factor(Scope, levels = c("All cells", "Epithelial")),
    Model = factor(Model, levels = c("PCA", "scVI", "scGPT", "Geneformer")),
    X_label = paste(Scope, Model, sep = "\n")
  )

model_colors <- c(
  "PCA" = "#66C2A5",
  "scVI" = "#FC8D62",
  "scGPT" = "#8DA0CB",
  "Geneformer" = "#D783C2"
)

strength_df <- df %>%
  select(Scope, Model, X_label, Max_abs_SMD, Mean_abs_SMD, Top10_signal_share) %>%
  pivot_longer(
    cols = c(Max_abs_SMD, Mean_abs_SMD, Top10_signal_share),
    names_to = "Metric",
    values_to = "Value"
  ) %>%
  mutate(
    Metric = recode(
      Metric,
      "Max_abs_SMD" = "Max abs SMD",
      "Mean_abs_SMD" = "Mean abs SMD",
      "Top10_signal_share" = "Top-10 signal share"
    ),
    Metric = factor(
      Metric,
      levels = c("Max abs SMD", "Mean abs SMD", "Top-10 signal share")
    )
  )

threshold_df <- df %>%
  select(Scope, Model, Dims_SMD_0_2, Dims_SMD_0_5) %>%
  pivot_longer(
    cols = c(Dims_SMD_0_2, Dims_SMD_0_5),
    names_to = "Threshold",
    values_to = "Dimensions"
  ) %>%
  mutate(
    Threshold = recode(
      Threshold,
      "Dims_SMD_0_2" = "abs SMD >= 0.2",
      "Dims_SMD_0_5" = "abs SMD >= 0.5"
    ),
    Threshold = factor(
      Threshold,
      levels = c("abs SMD >= 0.2", "abs SMD >= 0.5")
    )
  )

p_strength <- ggplot(strength_df, aes(x = X_label, y = Value, fill = Model)) +
  geom_col(
    width = 0.68,
    color = "black",
    linewidth = 0.25
  ) +
  geom_text(
    aes(label = sprintf("%.3f", Value)),
    vjust = -0.35,
    size = 2.7
  ) +
  facet_wrap(~ Metric, nrow = 1) +
  scale_fill_manual(values = model_colors) +
  scale_y_continuous(
    limits = c(0, 1.05),
    breaks = seq(0, 1.0, 0.25),
    expand = expansion(mult = c(0, 0.08))
  ) +
  labs(
    title = "A. Embedding disease-separation strength",
    x = NULL,
    y = "Value"
  ) +
  theme_classic(base_size = 12) +
  theme(
    plot.title = element_text(face = "bold", size = 15),
    legend.position = "none",
    strip.background = element_rect(fill = "grey92", color = "grey60"),
    strip.text = element_text(face = "bold", size = 10),
    axis.text.x = element_text(angle = 35, hjust = 1, size = 8.2),
    axis.text.y = element_text(size = 9),
    axis.title.y = element_text(face = "bold"),
    panel.spacing = unit(0.7, "lines")
  )

p_dims_02 <- threshold_df %>%
  filter(Threshold == "abs SMD >= 0.2") %>%
  ggplot(aes(x = Model, y = Dimensions, fill = Model)) +
  geom_col(width = 0.68, color = "black", linewidth = 0.25) +
  geom_text(aes(label = Dimensions), vjust = -0.35, size = 3.0) +
  facet_wrap(~ Scope, nrow = 1) +
  scale_fill_manual(values = model_colors) +
  scale_y_continuous(
    limits = c(0, 240),
    breaks = seq(0, 240, 40),
    expand = expansion(mult = c(0, 0.08))
  ) +
  labs(
    title = "B. Dimensions with abs SMD >= 0.2",
    x = NULL,
    y = "Dimensions"
  ) +
  theme_classic(base_size = 12) +
  theme(
    plot.title = element_text(face = "bold", size = 15),
    legend.position = "none",
    strip.background = element_rect(fill = "grey92", color = "grey60"),
    strip.text = element_text(face = "bold", size = 10),
    axis.text.x = element_text(angle = 35, hjust = 1, size = 9),
    axis.text.y = element_text(size = 9),
    axis.title.y = element_text(face = "bold")
  )

p_dims_05 <- threshold_df %>%
  filter(Threshold == "abs SMD >= 0.5") %>%
  ggplot(aes(x = Model, y = Dimensions, fill = Model)) +
  geom_col(width = 0.68, color = "black", linewidth = 0.25) +
  geom_text(aes(label = Dimensions), vjust = -0.35, size = 3.0) +
  facet_wrap(~ Scope, nrow = 1) +
  scale_fill_manual(values = model_colors) +
  scale_y_continuous(
    limits = c(0, 15),
    breaks = seq(0, 15, 5),
    expand = expansion(mult = c(0, 0.12))
  ) +
  labs(
    title = "C. Dimensions with abs SMD >= 0.5",
    x = NULL,
    y = "Dimensions"
  ) +
  theme_classic(base_size = 12) +
  theme(
    plot.title = element_text(face = "bold", size = 15),
    legend.position = "none",
    strip.background = element_rect(fill = "grey92", color = "grey60"),
    strip.text = element_text(face = "bold", size = 10),
    axis.text.x = element_text(angle = 35, hjust = 1, size = 9),
    axis.text.y = element_text(size = 9),
    axis.title.y = element_text(face = "bold")
  )

final_plot <- p_strength / p_dims_02 / p_dims_05 +
  plot_layout(heights = c(1.25, 1, 1))

print(final_plot)

ggsave(
  filename = "embedding_disease_separation_smd_summary_fully_labeled.png",
  plot = final_plot,
  width = 12,
  height = 12,
  dpi = 300,
  bg = "white"
)