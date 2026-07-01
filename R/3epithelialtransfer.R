install.packages(c("ggplot2", "RColorBrewer"))

library(ggplot2)
library(RColorBrewer)

transfer_df <- data.frame(
  model = c(
    "scVI", "PCA", "scVI", "scVI", "PCA",
    "PCA", "scVI", "scVI", "scVI", "scVI",
    "scVI", "PCA", "scVI", "PCA", "PCA"
  ),
  transfer = c(
    "Kong -> Garrido",
    "Kong -> Smillie",
    "Kong -> Smillie",
    "Garrido + Kong -> Smillie",
    "Garrido + Kong -> Smillie",
    "Garrido -> Smillie",
    "Garrido -> Smillie",
    "Kong + Smillie -> Garrido",
    "Smillie -> Garrido",
    "Smillie -> Kong",
    "Garrido + Smillie -> Kong",
    "Smillie -> Garrido",
    "Garrido -> Kong",
    "Kong -> Garrido",
    "Kong + Smillie -> Garrido"
  ),
  auroc = c(
    0.862,
    0.851,
    0.835,
    0.826,
    0.812,
    0.810,
    0.807,
    0.795,
    0.779,
    0.775,
    0.759,
    0.753,
    0.750,
    0.720,
    0.712
  )
)

transfer_df$label <- paste0(transfer_df$model, " | ", transfer_df$transfer)

transfer_df$label <- factor(
  transfer_df$label,
  levels = rev(transfer_df$label)
)

transfer_df$model <- factor(
  transfer_df$model,
  levels = c("PCA", "scVI")
)

p_transfer <- ggplot(transfer_df, aes(x = auroc, y = label, fill = model)) +
  geom_col(
    width = 0.7,
    color = "black",
    linewidth = 0.2
  ) +
  geom_text(
    aes(label = sprintf("%.3f", auroc)),
    hjust = -0.15,
    size = 3.5,
    family = "Arial"
  ) +
  scale_fill_brewer(palette = "Set2") +
  scale_x_continuous(
    limits = c(0, 0.95),
    breaks = seq(0, 0.9, 0.1),
    expand = expansion(mult = c(0, 0.05))
  ) +
  labs(
    title = "Ranked cross-dataset transfer AUROC benchmark",
    x = "AUROC",
    y = NULL,
    fill = "Model"
  ) +
  theme_classic(base_size = 12) +
  theme(
    legend.position = "top",
    legend.title = element_text(face = "bold"),
    axis.text.y = element_text(size = 10),
    plot.margin = margin(t = 15, r = 25, b = 15, l = 10)
  )

print(p_transfer)

ggplot2::ggsave(
  filename = "ranked_cross_dataset_transfer_auroc.png",
  plot = p_transfer,
  width = 10,
  height = 7,
  dpi = 300
)