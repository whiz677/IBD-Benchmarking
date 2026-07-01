library(ggplot2)
library(dplyr)
library(tidyr)

df <- tibble::tribble(
  ~Model, ~Dataset, ~AUROC, ~AUPRC, ~Accuracy,
  "scVI",       "Overall", 0.655, 0.537, 0.567,
  "scVI",       "Martin",  0.720, 0.827, 0.675,
  "scVI",       "Oliver",  0.656, 0.498, 0.549,
  
  "scGPT",      "Overall", 0.550, 0.446, 0.514,
  "scGPT",      "Martin",  0.526, 0.682, 0.531,
  "scGPT",      "Oliver",  0.562, 0.414, 0.511,
  
  "Geneformer", "Overall", 0.609, 0.614, 0.586,
  "Geneformer", "Martin",  0.646, 0.799, 0.584,
  "Geneformer", "Oliver",  0.618, 0.605, 0.586
)

plot_df <- df %>%
  mutate(
    Model = factor(Model, levels = c("scVI", "scGPT", "Geneformer")),
    Dataset = factor(Dataset, levels = c("Overall", "Martin", "Oliver"))
  ) %>%
  pivot_longer(
    cols = c(AUROC, AUPRC, Accuracy),
    names_to = "Metric",
    values_to = "Value"
  ) %>%
  mutate(
    Metric = factor(Metric, levels = c("AUROC", "AUPRC", "Accuracy"))
  )

p <- ggplot(plot_df, aes(x = Dataset, y = Value, fill = Model)) +
  geom_col(
    position = position_dodge(width = 0.72),
    width = 0.62,
    color = "black",
    linewidth = 0.25
  ) +
  geom_text(
    aes(label = sprintf("%.3f", Value)),
    position = position_dodge(width = 0.72),
    vjust = -0.35,
    size = 2.3
  ) +
  facet_wrap(~ Metric, nrow = 1) +
  scale_fill_manual(
    values = c(
      "scVI" = "#8DA0CB",
      "scGPT" = "#FC8D62",
      "Geneformer" = "#66C2A5"
    )
  ) +
  scale_y_continuous(
    limits = c(0, 0.90),
    breaks = seq(0, 0.9, 0.1),
    expand = expansion(mult = c(0, 0.04))
  ) +
  labs(
    title = "Fine-tuned all-cell locked external benchmark",
    x = NULL,
    y = "Performance",
    fill = "Model"
  ) +
  guides(
    fill = guide_legend(
      title.position = "top",
      title.hjust = 0.5,
      nrow = 1
    )
  ) +
  theme_classic(base_size = 13) +
  theme(
    plot.title = element_text(face = "bold", size = 18, hjust = 0),
    
    legend.position = "top",
    legend.title = element_text(face = "bold", size = 13),
    legend.text = element_text(size = 11),
    legend.key.size = unit(0.45, "cm"),
    
    strip.background = element_rect(fill = "grey92", color = "grey60"),
    strip.text = element_text(face = "bold", size = 11),
    
    axis.text.x = element_text(angle = 35, hjust = 1, size = 10),
    axis.text.y = element_text(size = 10),
    axis.title.y = element_text(face = "bold", size = 12),
    
    plot.margin = margin(15, 25, 15, 25)
  )

print(p)

ggsave(
  filename = "figure8_finetuned_all_cell_locked_external_benchmark.png",
  plot = p,
  width = 11,
  height = 5.2,
  dpi = 300,
  bg = "white"
)