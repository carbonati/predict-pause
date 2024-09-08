#!/usr/bin/env Rscript

print(paste("Starting input duration experiment @", Sys.time()))

pkgs <- c("pROC", "ggpubr", "PRROC")
new.pkgs <- pkgs[!(pkgs %in% installed.packages()[, "Package"])]
if (length(new.pkgs)) install.packages(new.pkgs, repos = "http://cran.us.r-project.org")

library("pROC")
library("PRROC")
library("ggpubr")

get_pval <- function(x) {
  if (x > 0.05) {
    return ("ns")
  } else if (x <= 0.05 & x > 0.01) {
    return ("*")
  } else if (x <= 0.01 & x > 0.001) {
    return ("**")
  } else if (x <= 0.001 & x > 0.0001) {
    return ("***")
  } else {
    return ("****")
  }
}

compute_input_duration_delong_test <- function(df_duration_preds, target_name, num_boot) {
  data <- c(
    y = c(),
    group1 = c(),
    group2 = c(),
    p = c(),
    p.signif = c()
  )
  for(i in 1:length(durations)) {
    data$y[i] <- "auc"
    data$group1[i] <- "all" # hotfix for the ggpubr package
    data$group2[i] <- durations[i]
    df_a <- df_duration_preds[
      (
        (df_duration_preds["duration"] == "24 h")
        & !is.na(df_duration_preds[, target_name])
      ),
    ]
    df_b <- df_duration_preds[
      (
        (df_duration_preds["duration"] == durations[i])
        & !is.na(df_duration_preds[, target_name])
      ),
    ]
    roc_a <- roc(df_a[, target_name], df_a[, paste0(target_name, "_pred")])
    roc_b <- roc(df_b[, target_name], df_b[, paste0(target_name, "_pred")])
    stat.test <- roc.test(roc_a, roc_b, method = "delong", boot.n = num_boot)

    data$p[i] <- stat.test$p.value
    data$p.signif[i] <- get_pval(stat.test$p.value)
  }
  df_stat <- do.call(data.frame, data)
  return (df_stat)
}

compute_paired_prc <- function(y_true, y_pred_a, y_pred_b, stratified) {
  if (stratified) {
    pos_indices = which(y_true == 1)
    neg_indices = which(y_true == 0)
    indices = c(sample(pos_indices, replace = TRUE), sample(neg_indices, replace = TRUE))
  } else {
    indices = sample(1:length(y_true), replace = TRUE)
  }

  # compute resampled paired AUPRCs
  y_true_resampled = y_true[indices]
  y_pred_a_resampled  = y_pred_a[indices]
  y_pred_b_resampled  = y_pred_b[indices]

  auc_a = pr.curve(
    scores.class0 = y_pred_a_resampled,
    weights.class0 = y_true_resampled
  )$auc.davis.goadrich
  auc_b = pr.curve(
    scores.class0 = y_pred_b_resampled,
    weights.class0 = y_true_resampled
  )$auc.davis.goadrich

  return (auc_a-auc_b)
}

compute_bootstrap_prc_test <- function(y_true, y_pred_a, y_pred_b, num_boot = 1000, stratified = TRUE) {
  auc_diffs <- sapply(
    1:num_boot,
    function(x) compute_paired_prc(y_true, y_pred_a, y_pred_b, FALSE),
    simplify = TRUE
  )

  auc_a = pr.curve(scores.class0 = y_pred_a, weights.class0 = y_true)$auc.davis.goadrich
  auc_b = pr.curve(scores.class0 = y_pred_b, weights.class0 = y_true)$auc.davis.goadrich

  # compute 2-sided p-value
  stat = (auc_a - auc_b) / sd(auc_diffs)
  pval = 2 * pnorm(-abs(stat))
  return (pval)
}

compute_input_duration_prc_test <- function(df_duration_preds, target_name, num_boot = 1000) {
  data <- c(
    y = c(),
    group1 = c(),
    group2 = c(),
    p = c(),
    p.signif = c()
  )
  for(i in 1:length(durations)) {
    data$y[i] <- "prc"
    data$group1[i] <- "all" # hotfix for the ggpubr package
    data$group2[i] <- durations[i]

    df_a <- df_duration_preds[
      (
        (df_duration_preds["duration"] == "24 h")
        & !is.na(df_duration_preds[, target_name])
      ),
    ]
    df_b <- df_duration_preds[
      (
        (df_duration_preds["duration"] == durations[i])
        & !is.na(df_duration_preds[, target_name])
      ),
    ]

    p.value <- compute_bootstrap_prc_test(
      df_a[, target_name],
      df_a[, paste0(target_name, "_pred")],
      df_b[, paste0(target_name, "_pred")],
      num_boot
    )

    data$p[i] <- p.value
    data$p.signif[i] <- get_pval(p.value)
  }
  df_stat <- do.call(data.frame, data)
  return (df_stat)
}

################################################################################
# CONSTANTS
################################################################################

target_name_map = c(
  retro_DAY_PAUSE_14d = "Daytime sinus pause ≥3",
  retro_LONG_PAUSE_14d = "Sinus pause ≥6",
  retro_AVB3_14d = "Complete AV block",
  retro_ASYSTOLE_14d = "Composite"
)
target_names <- names(target_name_map)
num_boot <- 1000
durations <- c("1 h", "2 h", "3 h", "4 h", "6 h", "12 h")
palette <- c("#3a93c3", "#d75f4c")
set.seed(42069)

args <- commandArgs(trailingOnly=TRUE)
if (length(args) == 2) {
	duration_preds_filepath <- args[1]
	duration_scores_filepath <- args[2]
	ouput_dir <- NULL
} else if (length(args) == 3) {
	duration_preds_filepath <- args[1]
	duration_scores_filepath <- args[2]
	ouput_dir <- args[3]
} else {
  stop(
    "\n\n  Expected 2 required argument and 1 optional argument.",
    "\n\n  Required arguments:",
    "\n\n      [1] Path to .csv file with model predictions by input duration",
    "\n\n      [2] Path to .csv file with metric scores by input duration",
    "\n\n  Optional arguments:",
    "\n\n      [3] Output directory to save forest plot, defaults to not saving plot (optional)\n"
  )
}

################################################################################
# LOAD TABLES
################################################################################

df_duration_preds <- read.csv(duration_preds_filepath)
df_duration_scores <- read.csv(duration_scores_filepath)

for (target_name in target_names) {
  print(paste("Generating input duration ROC plot for", target_name))
  df_target_scores <- df_duration_scores[df_duration_scores["target_name"] == target_name, ]
  df_target_scores$partition <- factor(df_target_scores$partition, level=c("Internal validation", "External validation"))

  ################################################################################
  # ROC FIGURE
  ################################################################################

  # delong ROC test
  df_stat <- compute_input_duration_delong_test(
    df_duration_preds[df_duration_preds["partition"] == "External validation", ],
    target_name,
    num_boot
  )

  # ROC input duratino plot
  p <- ggboxplot(
    df_target_scores,
    x = "duration",
    y = "auc",
    color = "partition",
    xlab = "Monitoring duration",
    ylab = "AUROC",
    palette = palette,
    size = 0.75,
    add.params = list(size = 5),
    bxp.errorbar = TRUE,
    bxp.errorbar.width = 0.25,
    outlier.shape = NA
  ) +
  ylim(0.75, 1) +
  scale_y_continuous(expand = c(0, 0), limits = c(0.75, 1)) +
  ggtitle(target_name_map[[target_name]]) +
  scale_colour_manual(
    name = NULL,
    values = palette,
    labels = c("Internal validation", "External validation")
  ) +
  theme(
    plot.title = element_text(hjust = 0.5, size=15),
    text = element_text(size = 12),
    legend.position = c(0.85, 0.1),
    legend.text = element_text(size=15),
    axis.ticks.x = element_blank(),
  ) +
  stat_pvalue_manual(
    df_stat,
    # y.position = max(df_target_scores$auc) + 0.01,
    y.position <- 0.98,
    label = "p.signif",
    size=5
  )

  for (i in 1:6) {
    p <- p + geom_vline(
      xintercept = i + 0.5,
      linetype = "dotted",
      color = "gray",
      size = 0.75,
      linewidth = 1,
      alpha = 1
    )
  }
  # plot(p)

  # save plot to forestplot to disk
  if ( is.character(output_dir) ) {
      # dev.off()
      plot_path <- file.path(output_dir, paste0(target_name, "_delong_auroc.png"))
      print(paste("Saving plot to", plot_path))
      dir.create(output_dir, showWarnings = FALSE)
      ggplot2::ggsave(filename = plot_path,
                      plot = p,
                      dpi = 300,
                      width = 8,
                      height = 6,
                      units = "in")
  }
  else {
      plot(p)
  }

  ################################################################################
  # PRC FIGURE
  ################################################################################

  print(paste("Generating input duration PRC plot for", target_name))

  # bootstrapped PRC test
  df_stat <- compute_input_duration_prc_test(
    df_duration_preds[df_duration_preds["partition"] == "External validation", ],
    target_name,
    num_boot
  )

  # ROC input duratino plot
  p <- ggboxplot(
    df_target_scores,
    x = "duration",
    y = "prc",
    color = "partition",
    xlab = "Monitoring duration",
    ylab = "AUPRC",
    palette = palette,
    size = 0.75,
    add.params = list(size = 5),
    bxp.errorbar = TRUE,
    bxp.errorbar.width = 0.25,
    outlier.shape = NA
  ) +
    ylim(0, 0.8) +
    scale_y_continuous(expand = c(0, 0), limits = c(0, 0.8)) +
    ggtitle(target_name_map[[target_name]]) +
    scale_colour_manual(
      name = NULL,
      values = palette,
      labels = c("Internal validation", "External validation")
    ) +
    theme(
      plot.title = element_text(hjust = 0.5, size=15),
      text = element_text(size = 12),
      legend.position = c(0.85, 0.1),
      legend.text = element_text(size=15),
      axis.ticks.x = element_blank(),
    ) +
    stat_pvalue_manual(
      df_stat,
      # y.position = max(df_target_scores$auc) + 0.01,
      y.position <- 0.75,
      label = "p.signif",
      size = 5
    )

  for (i in 1:6) {
    p <- p + geom_vline(
      xintercept = i + 0.5,
      linetype = "dotted",
      color = "gray",
      size = 0.75,
      linewidth = 1,
      alpha = 1
    )
  }

  if ( is.character(output_dir) ) {
      plot_path <- file.path(output_dir, paste0(target_name, "_bootstrap_prc.png"))
      print(paste("Saving plot to", plot_path))
      dir.create(output_dir, showWarnings = FALSE)
      ggplot2::ggsave(filename = plot_path,
                      plot = p,
                      dpi = 300,
                      width = 8,
                      height = 6,
                      units = "in")
  }
  else {
      plot(p)
  }
}

print(paste("Finished input duration experiment @", Sys.time()))
