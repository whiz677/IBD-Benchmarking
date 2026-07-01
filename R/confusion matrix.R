packages <- c("ggplot2", "scales")

for (pkg in packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg)
  }
}

library(ggplot2)
library(scales)

source_file <- "C:/Users/activ/OneDrive/Pictures/scDiseaseShift_IBD_starter_FULL_PROJECT/scDiseaseShift_IBD_starter/results_final6/proper_no_leak_benchmark/manual_registry_no_leak_disease_metrics.csv"

out_dir <- "C:/Users/activ/OneDrive/Documents/ML R2/ibd_manuscript_outputs/figures"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

df <- read.csv(source_file, stringsAsFactors = FALSE)

df <- df[df$test_type == "locked_external", ]

df$model_clean <- df$model
df$model_clean[df$model == "PCA_50"] <- "PCA"
df$model_clean[df$model == "scVI_30"] <- "scVI"
df$model_clean[df$model == "scGPT_frozen"] <- "scGPT"
df$model_clean[df$model == "Geneformer_frozen"] <- "Geneformer"

df$scope_clean <- df$group
df$scope_clean[df$group == "all_cells"] <- "All cells"
df$scope_clean[df$group == "epithelial"] <- "Epithelial"

df$test_clean <- df$test_dataset
df$test_clean[df$test_dataset == "martin"] <- "Martin"
df$test_clean[df$test_dataset == "oliver_no_martin"] <- "Oliver-no-Martin"

df$setting <- paste(df$scope_clean, df$test_clean, sep = " | ")

model_levels <- c("PCA", "scVI", "scGPT", "Geneformer")
setting_levels <- c(
  "All cells | Martin",
  "All cells | Oliver-no-Martin",
  "Epithelial | Martin",
  "Epithelial | Oliver-no-Martin"
)

df$model_clean <- factor(df$model_clean, levels = model_levels)
df$setting <- factor(df$setting, levels = setting_levels)

confusion_df <- rbind(
  data.frame(
    setting = df$setting,
    model = df$model_clean,
    actual = "Control",
    predicted = "Control",
    count = df$tn
  ),
  data.frame(
    setting = df$setting,
    model = df$model_clean,
    actual = "Control",
    predicted = "IBD",
    count = df$fp
  ),
  data.frame(
    setting = df$setting,
    model = df$model_clean,
    actual = "IBD",
    predicted = "Control",
    count = df$fn
  ),
  data.frame(
    setting = df$setting,
    model = df$model_clean,
    actual = "IBD",
    predicted = "IBD",
    count = df$tp
  )
)

confusion_df$actual <- factor(confusion_df$actual, levels = c("IBD", "Control"))
confusion_df$predicted <- factor(confusion_df$predicted, levels = c("Control", "IBD"))
confusion_df$model <- factor(confusion_df$model, levels = model_levels)
confusion_df$setting <- factor(confusion_df$setting, levels = setting_levels)

confusion_df$count_label <- comma(confusion_df$count)

p <- ggplot(
  confusion_df,
  aes(x = predicted, y = actual, fill = count)
) +
  geom_tile(color = "white", linewidth = 1.1) +
  geom_text(
    aes(label = count_label),
    size = 3.4,
    fontface = "bold",
    color = "black"
  ) +
  facet_grid(setting ~ model) +
  scale_fill_gradient(
    low = "#f7fbff",
    high = "#08519c",
    trans = "sqrt",
    labels = comma,
    name = "Count"
  ) +
  labs(
    title = "Locked external confusion-matrix counts",
    subtitle = "Positive class = IBD; negative class = control",
    x = "Predicted label",
    y = "Actual label"
  ) +
  coord_fixed() +
  theme_minimal(base_size = 12) +
  theme(
    plot.title = element_text(face = "bold", size = 16, hjust = 0.5),
    plot.subtitle = element_text(size = 11, hjust = 0.5),
    strip.text.x = element_text(face = "bold", size = 11),
    strip.text.y = element_text(face = "bold", size = 10, angle = 0),
    axis.text.x = element_text(size = 10),
    axis.text.y = element_text(size = 10),
    axis.title.x = element_text(face = "bold"),
    axis.title.y = element_text(face = "bold"),
    panel.grid = element_blank(),
    legend.title = element_text(face = "bold"),
    plot.margin = margin(t = 15, r = 15, b = 15, l = 15)
  )

print(p)

ggplot2::ggsave(
  filename = file.path(out_dir, "locked_external_confusion_matrix_counts.png"),
  plot = p,
  width = 12,
  height = 9,
  dpi = 300
)