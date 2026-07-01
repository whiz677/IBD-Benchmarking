# =========================
# Training matrix figure
# All-cell vs epithelial-only training composition
# =========================

library(ggplot2)
library(dplyr)
library(tidyr)

df <- tibble::tribble(
  ~Training_set, ~Dataset, ~Class, ~Cells,
  "All-cell training", "Garrido-Trigo",      "Healthy/control", 12000,
  "All-cell training", "Garrido-Trigo",      "IBD",             12000,
  "All-cell training", "Kong",               "Healthy/control", 12000,
  "All-cell training", "Kong",               "IBD",             12000,
  "All-cell training", "Smillie epithelial", "Healthy/control", 12000,
  "All-cell training", "Smillie epithelial", "IBD",             12000,
  
  "Epithelial-only training", "Garrido-Trigo epithelial", "Healthy/control", 5287,
  "Epithelial-only training", "Garrido-Trigo epithelial", "IBD",             2064,
  "Epithelial-only training", "Kong epithelial",          "Healthy/control", 4471,
  "Epithelial-only training", "Kong epithelial",          "IBD",             4020,
  "Epithelial-only training", "Smillie epithelial",       "Healthy/control", 12000,
  "Epithelial-only training", "Smillie epithelial",       "IBD",             12000
)

totals <- df %>%
  group_by(Training_set, Dataset) %>%
  summarise(Total = sum(Cells), .groups = "drop") %>%
  mutate(label = paste0(format(Total, big.mark = ","), " cells"))

df <- df %>%
  mutate(
    Training_set = factor(
      Training_set,
      levels = c("All-cell training", "Epithelial-only training")
    ),
    Class = factor(Class, levels = c("Healthy/control", "IBD"))
  )

p <- ggplot(df, aes(x = Dataset, y = Cells, fill = Class)) +
  geom_col(
    width = 0.68,
    color = "black",
    linewidth = 0.25
  ) +
  geom_text(
    data = totals,
    aes(x = Dataset, y = Total, label = label),
    inherit.aes = FALSE,
    vjust = -0.45,
    size = 3.8,
    fontface = "bold"
  ) +
  facet_wrap(~ Training_set, nrow = 1, scales = "free_x") +
  scale_fill_manual(
    values = c(
      "Healthy/control" = "#7FCDBB",
      "IBD" = "#F46D43"
    )
  ) +
  scale_y_continuous(
    labels = scales::comma,
    expand = expansion(mult = c(0, 0.12))
  ) +
  labs(
    title = "Training-cell composition for locked external benchmarks",
    x = NULL,
    y = "Number of training cells",
    fill = "Class"
  ) +
  theme_classic(base_size = 15) +
  theme(
    plot.title = element_text(face = "bold", size = 19),
    legend.position = "top",
    legend.title = element_text(face = "bold"),
    strip.background = element_rect(fill = "grey92", color = "grey60"),
    strip.text = element_text(face = "bold", size = 14),
    axis.text.x = element_text(angle = 30, hjust = 1),
    axis.title.y = element_text(face = "bold"),
    plot.margin = margin(18, 25, 18, 25)
  )

print(p)

ggsave(
  filename = "training_cell_composition_two_panel.png",
  plot = p,
  width = 11,
  height = 5.2,
  dpi = 300,
  bg = "white"
)