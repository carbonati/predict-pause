#!/usr/bin/env Rscript

print(paste("Starting subgroup analysis script @", Sys.time()))

# install packages
pkgs <- c("devtools", "forestploter", "forester")
new.pkgs <- pkgs[!(pkgs %in% installed.packages()[, "Package"])]
if (length(new.pkgs)) install.packages(new.pkgs, repos = "http://cran.us.r-project.org")

# install git
git.pkgs <- "rdboyes/forester"
new.pkgs <- pkgs[!(pkgs %in% installed.packages()[, "Package"])]
if (length(new.pkgs)) devtools::install_github(git.pkgs)

library("forestploter")
library("grid")

args <- commandArgs(trailingOnly=TRUE)
if (length(args) == 1) {
  input_filepath <- args[1]
  run_id <- NULL
} else if (length(args) == 2) {
  input_filepath <- args[1]
  output_dir <- args[2]
} else {
  stop(
    "\n\n  Expected 1 required argument and 1 optional argument.",
    "\n\n  Required arguments:",
    "\n\n      [1] Path to processed forestplot table with bootstrapped scores",
    "\n\n  Optional arguments:",
    "\n\n      [2] Output directory to save forest plot, defaults to not saving plot (optional)\n"
  )
}

# load preprocessed subgroup table
df <- read.csv(input_filepath, skip = 1, check.names = FALSE)
headers <- as.list(read.table(input_filepath, nrows = 1, sep = ",", header = FALSE)[1, ])
names(df) <- c(c("Subgroup"), paste0(headers[2:length(df)], "-", names(df)[2:length(df)]))

# simplify some subgroup names
df[df$Subgroup == "    Arrhythmia, SVT, or VT", "Subgroup"] <- "    Arrhythmia"

df$cicol <- paste(rep(" ", 20), collapse = " ") # spacing column
df$Subgroup <- paste0(df$Subgroup, "    ")

# prepare forest table
table_colnames <- c(
  "Subgroup",
  "cicol",
  "retro_DAY_PAUSE_14d-AUROC (95% CI)",
  "cicol",
  "retro_LONG_PAUSE_14d-AUROC (95% CI)",
  "cicol",
  "retro_AVB3_14d-AUROC (95% CI)",
  "cicol",
  "retro_ASYSTOLE_14d-AUROC (95% CI)"
)
df_forest <- df[, table_colnames]
colnames(df_forest) <- c("Subgroup", "", "AUROC (95% CI)", "", "AUROC (95% CI)", "", "AUROC (95% CI)", "", "AUROC (95% CI)")
head(df_forest)

# forestplot theme
tm <- forest_theme(
    base_size = 10,
    ci_pch = c(15, 15, 15, 15),
    legend_value = c("Daytime sinus pause ≥3s", "Sinus pause ≥6s", "Complete AV block", "Composite"),
    ci_col = c("#1A98D9", "#1A98D9", "#1A98D9", "#1A98D9"),
    ci_fill = c("#1A98D9", "#1A98D9", "#1A98D9", "#1A98D9"),
    ci_alpha = 0.8,
    ci_lwd = 1.5,
    ci_Theight = 0.3,
    refline_lty = "dashed",
    xlab_adjust = "center",
    summary_fill = "#4575b4",
    summary_col = "#4575b4",
    # core=list(bg_params=list(fill = c("white"))),
)


# rescale size of point estimates based on sample size
rescale <- function(x, a, b) {
  return ((b - a) * (x - min(x, na.rm = TRUE)) / (max(x, na.rm = TRUE) - min(x, na.rm = TRUE)) + a)
}

n <- 42173
df[, "retro_DAY_PAUSE_14d-size"] <- rescale(df[, c("retro_DAY_PAUSE_14d-N")] / n, 0.33, 1)
df[, "retro_LONG_PAUSE_14d-size"] <- rescale(df[, c("retro_LONG_PAUSE_14d-N")] / n, 0.33, 1)
df[, "retro_AVB3_14d-size"] <- rescale(df[, c("retro_AVB3_14d-N")] / n, 0.33, 1)
df[, "retro_ASYSTOLE_14d-size"] <- rescale(df[, c("retro_ASYSTOLE_14d-N")] / n, 0.33, 1)

# forestplot
p <- forest(
    df_forest,
    est = list(
        df$`retro_DAY_PAUSE_14d-AUROC`,
        df$`retro_LONG_PAUSE_14d-AUROC`,
        df$`retro_AVB3_14d-AUROC`,
        df$`retro_ASYSTOLE_14d-AUROC`
    ),
    lower = list(
        df$`retro_DAY_PAUSE_14d-AUROC_low`,
        df$`retro_LONG_PAUSE_14d-AUROC_low`,
        df$`retro_AVB3_14d-AUROC_low`,
        df$`retro_ASYSTOLE_14d-AUROC_low`
    ),
    upper = list(
        df$`retro_DAY_PAUSE_14d-AUROC_high`,
        df$`retro_LONG_PAUSE_14d-AUROC_high`,
        df$`retro_AVB3_14d-AUROC_high`,
        df$`retro_ASYSTOLE_14d-AUROC_high`
    ),
    sizes=list(
        df[, c("retro_DAY_PAUSE_14d-size")],
        df[, c("retro_LONG_PAUSE_14d-size")],
        df[, c("retro_AVB3_14d-size")],
        df[, c("retro_ASYSTOLE_14d-size")]
    ),
    ci_column = c(2, 4, 6, 8),
    ref_line = c(
        df[df$Subgroup == "Overall    ", ]$`retro_DAY_PAUSE_14d-AUROC`,
        df[df$Subgroup == "Overall    ", ]$`retro_LONG_PAUSE_14d-AUROC`,
        df[df$Subgroup == "Overall    ", ]$`retro_AVB3_14d-AUROC`,
        df[df$Subgroup == "Overall    ", ]$`retro_ASYSTOLE_14d-AUROC`
    ),
    xlim = list(c(0.79, 1), c(0.7, 1), c(0.64, 1), c(0.8, 1)),
    ticks_at = list(c(0.8, 0.9, 1), c(0.7, 0.8, 0.9, 1), c(0.7, 0.8, 0.9, 1), c(0.8, 0.9, 1)),
    xlab = "AUROC",
    theme = tm
)

# add headers
g <- add_text(p,
              text = "Daytime sinus pause ≥3s",
              col = 1:3,
              row = 0,
              part = "header",
              gp = gpar(fontface = "bold"))
g <- add_text(g,
              text = "Sinus pause ≥6s",
              col = 3:5,
              row = 0,
              part = "header",
              gp = gpar(fontface = "bold"))
g <- add_text(g,
              text = "Complete AV block",
              row = 0,
              col = 5:7,
              part = "header",
              gp = gpar(fontface = "bold"))
g <- add_text(g,
              text = "Composite",
              row = 0,
              col = 7:9,
              part = "header",
              gp = gpar(fontface = "bold"))

# indent groups
g <- edit_plot(g,
               row = c(1, 4, 7, 10, 13, 16, 19, 27),
               col = 1,
               gp = gpar(fontface = "bold"))

# save plot to forestplot to disk
if ( is.character(output_dir) ) {
  dev.off()
  plot_path <- file.path(output_dir, "predict_pause_subgroup_analysis_auroc.png")
  print(paste("Saving forest plot to", plot_path))
  ggplot2::ggsave(filename = plot_path,
                  plot = g,
                  dpi = 300,
                  width = 13.5,
                  height = 7,
                  units = "in")
  ggplot2::ggsave(filename = file.path(output_dir, "predict_pause_subgroup_analysis_auroc.eps"),
                  plot = g,
                  dpi = 300,
                  width = 13.5,
                  height = 7,
                  units = "in")
}
else {
  plot(g)
}

print(paste("Finished subgroup analysis script @", Sys.time()))
