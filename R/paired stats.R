# =========================
# Donor-aware paired AUROC comparison forest plot
# Clean display version: labels above lines, free x-axis per panel
# =========================

library(ggplot2)
library(dplyr)
library(stringr)

bootstrap_file <- "C:/Users/activ/OneDrive/Documents/ML R2/ibd_external_stats/locked_external_paired_model_comparisons_donor_bootstrap.csv"

bootstrap <- read.csv(bootstrap_file, stringsAsFactors = FALSE)

clean_model <- function(x) {
  x %>%
    str_replace("_50$", "") %>%
    str_replace("_30$", "") %>%
    str_replace("_frozen$", "")
}

clean_setting <- function(group, test_dataset) {
  case_when(
    group == "all_cells" & test_dataset == "martin" ~ "Martin all cells",
    group == "epithelial" & test_dataset == "martin" ~ "Martin epithelial",
    group == "all_cells" & test_dataset == "oliver_no_martin" ~ "Oliver-no-Martin all cells",
    group == "epithelial" & test_dataset == "oliver_no_martin" ~ "Oliver-no-Martin epithelial",
    TRUE ~ paste(group, test_dataset)
  )
}

format_fdr <- function(x) {
  case_when(
    x == 0 ~ "FDR < 1e-300",
    x < 0.001 ~ paste0("FDR = ", formatC(x, format = "e", digits = 1)),
    TRUE ~ paste0("FDR = ", sprintf("%.3f", x))
  )
}

plot_df <- bootstrap %>%
  mutate(
    Setting = clean_setting(group, test_dataset),
    Comparison = paste(clean_model(model_a), "vs", clean_model(model_b)),
    Delta = delta_auroc_a_minus_b,
    CI_low = delta_auroc_ci_low,
    CI_high = delta_auroc_ci_high,
    FDR = auroc_bh_fdr,
    FDR_label = format_fdr(FDR),
    Significant = ifelse(FDR < 0.05, "FDR < 0.05", "ns"),
    Row_label = paste0(Comparison, "\n", FDR_label)
  ) %>%
  mutate(
    Setting = factor(
      Setting,
      levels = c(
        "Martin all cells",
        "Oliver-no-Martin all cells",
        "Martin epithelial",
        "Oliver-no-Martin epithelial"
      )
    ),
    Comparison_order = factor(
      Comparison,
      levels = rev(c(
        "PCA vs Geneformer",
        "PCA vs scGPT",
        "scVI vs scGPT",
        "PCA vs scVI"
      ))
    )
  ) %>%
  arrange(Setting, Comparison_order) %>%
  group_by(Setting) %>%
  mutate(
    y_position = row_number()
  ) %>%
  ungroup()

p <- ggplot(
  plot_df,
  aes(
    x = Delta,
    y = y_position
  )
) +
  geom_vline(
    xintercept = 0,
    linetype = "dashed",
    color = "grey45",
    linewidth = 0.55
  ) +
  geom_errorbarh(
    aes(
      xmin = CI_low,
      xmax = CI_high,
      color = Significant
    ),
    height = 0.16,
    linewidth = 0.85
  ) +
  geom_point(
    aes(color = Significant),
    size = 3.0
  ) +
  geom_text(
    aes(
      y = y_position + 0.18,
      label = sprintf("%.3f", Delta)
    ),
    size = 3.0,
    color = "black",
    show.legend = FALSE
  ) +
  facet_wrap(~ Setting, ncol = 2, scales = "free_x") +
  scale_y_continuous(
    breaks = plot_df$y_position,
    labels = plot_df$Row_label,
    expand = expansion(mult = c(0.15, 0.22))
  ) +
  scale_color_manual(
    values = c(
      "FDR < 0.05" = "#2B8CBE",
      "ns" = "grey55"
    )
  ) +
  labs(
    title = "Donor-aware paired AUROC comparisons",
    x = "Delta AUROC",
    y = NULL
  ) +
  theme_classic(base_size = 13) +
  theme(
    plot.title = element_text(face = "bold", size = 17),
    
    legend.position = "none",
    
    strip.background = element_rect(fill = "grey92", color = "grey60"),
    strip.text = element_text(face = "bold", size = 12),
    
    axis.text.y = element_text(size = 8.6, lineheight = 0.9),
    axis.text.x = element_text(size = 10),
    axis.title.x = element_text(face = "bold"),
    
    panel.spacing = unit(1.8, "lines"),
    plot.margin = margin(15, 25, 15, 25)
  )

print(p)

ggsave(
  filename = "paired_auroc_comparisons_donor_bootstrap_clean_display.png",
  plot = p,
  width = 12.5,
  height = 8.2,
  dpi = 300,
  bg = "white"
)