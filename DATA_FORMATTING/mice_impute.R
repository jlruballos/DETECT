# -------------------------------
# MICE Imputation for All Subjects – Combined Output + Plots
# -------------------------------

library(mice)
library(ggplot2)
library(dplyr)
library(tidyr)
library(data.table)
library(purrr)
library(zoo)

# -------- CONFIG --------
method_name <- "pmm"
input_path <- "D:/DETECT/OUTPUT/raw_export_for_r/raw_daily_data_all_subjects.csv"
output_root <- file.path("D:/DETECT/OUTPUT/raw_export_for_r", method_name)
plot_root <- file.path(output_root, "diagnostics")
output_file <- file.path(output_root, paste0("labeled_daily_data_", method_name, "_all_subjects.csv"))

dir.create(output_root, showWarnings = FALSE, recursive = TRUE)
dir.create(plot_root, showWarnings = FALSE, recursive = TRUE)

# -------- Load & Clean Data --------
df <- read.csv(input_path)
df$date <- as.Date(df$date)

# Drop engineered columns
engineered_cols <- grep("(_delta|_norm|_ma_7|_delta_1d)", colnames(df), value = TRUE)
df <- df %>% select(-all_of(engineered_cols))

# Define base columns
exclude_cols <- c("subid", "date", "label_fall", "label_hospital", "label",
                  "days_since_fall", "days_until_fall", 
                  "days_since_hospital", "days_until_hospital")
feature_cols <- setdiff(colnames(df), exclude_cols)

# Add lag features
lag_days <- c(1, 7)
df <- df %>%
  arrange(subid, date) %>%
  group_by(subid) %>%
  mutate(across(all_of(feature_cols),
                .fns = setNames(lapply(lag_days, function(l) function(x) lag(x, l)),
                                paste0("lag", lag_days)),
                .names = "{.col}_{.fn}")) %>%
  ungroup()

lagged_feature_cols <- grep("_lag\\d+$", colnames(df), value = TRUE)
mice_feature_cols <- c(feature_cols, lagged_feature_cols)

# -------- Loop Through Subjects --------
all_subids <- unique(df$subid)
all_imputed <- list()

for (sid in all_subids) {
  cat("🔄 Processing subject:", sid, "\n")
  
  sub_df <- df %>% filter(subid == sid)
  if (nrow(sub_df) < 3) next
  
  missing_mask <- sub_df %>% select(all_of(feature_cols)) %>% is.na() %>% as.data.frame()
  
  imp <- mice(sub_df[, mice_feature_cols], m = 1, method = method_name, maxit = 5,
              seed = 42, printFlag = FALSE)
  completed <- complete(imp, 1)
  
  # Reattach metadata
  completed$subid <- sub_df$subid
  completed$date <- sub_df$date
  completed$label_fall <- sub_df$label_fall
  completed$label_hospital <- sub_df$label_hospital
  completed$label <- sub_df$label
  completed$days_since_fall <- sub_df$days_since_fall
  completed$days_until_fall <- sub_df$days_until_fall
  completed$days_since_hospital <- sub_df$days_since_hospital
  completed$days_until_hospital <- sub_df$days_until_hospital
  
  # Clamp & Flag
  for (col in feature_cols) {
    completed[[col]] <- pmax(completed[[col]], 0)
    completed[[paste0(col, "_imputed")]] <- as.integer(missing_mask[[col]])
  }
  
  # Temporal Features
  completed <- completed %>%
    arrange(date) %>%
    mutate(across(all_of(feature_cols), 
                  list(
                    delta = ~ . - mean(., na.rm = TRUE),
                    delta_1d = ~ c(NA, diff(.)),
                    norm = ~ {
                      m <- mean(., na.rm = TRUE)
                      s <- sd(., na.rm = TRUE)
                      ifelse(s > 0, (. - m)/s, NA)
                    },
                    ma_7 = ~ rollapply(., 7, mean, fill = NA, align = "right", partial = TRUE)
                  ), .names = "{.col}_{.fn}"))
  
  # Save diagnostic plots
  sub_dir <- file.path(plot_root, paste0("subid_", sid))
  dir.create(sub_dir, showWarnings = FALSE, recursive = TRUE)
  histo_dir <- file.path(sub_dir, "histograms")
  line_dir <- file.path(sub_dir, "lineplots")
  dir.create(histo_dir, showWarnings = FALSE)
  dir.create(line_dir, showWarnings = FALSE)
  
  for (col in feature_cols) {
    orig_vals <- sub_df[[col]]
    imputed_vals <- completed[[col]]
    imputed_flag <- missing_mask[[col]]
    
    if (all(is.na(orig_vals)) || all(is.na(imputed_vals))) next
    
    # Histogram
    overlay_df <- data.frame(
      value = c(orig_vals[!is.na(orig_vals)], imputed_vals[imputed_flag]),
      type = c(rep("Original", sum(!is.na(orig_vals))),
               rep("Imputed", sum(imputed_flag, na.rm = TRUE)))
    )
    
    g_hist <- ggplot(overlay_df, aes(x = value, fill = type)) +
      geom_histogram(alpha = 0.5, position = "identity", bins = 30) +
      scale_fill_manual(values = c("Original" = "gray", "Imputed" = "red")) +
      labs(title = paste("Histogram:", col, "- Subid", sid),
           x = col, y = "Count") +
      theme_minimal()
    ggsave(filename = file.path(histo_dir, paste0(col, "_hist_", sid, ".png")),
           plot = g_hist, width = 8, height = 4)
    
    # Time series
    plot_df <- completed
    plot_df$imputed <- imputed_flag
    
    g_line <- ggplot(plot_df, aes(x = date, y = !!sym(col))) +
      geom_line(color = "black") +
      geom_point(data = subset(plot_df, imputed == TRUE),
                 aes(x = date, y = !!sym(col)),
                 color = "red", size = 2) +
      labs(title = paste("Imputed:", col, "- Subid", sid),
           subtitle = paste(sum(imputed_flag, na.rm = TRUE), "points"),
           y = col, x = "Date") +
      theme_minimal()
    ggsave(filename = file.path(line_dir, paste0(col, "_line_", sid, ".png")),
           plot = g_line, width = 8, height = 4)
  }
  
  all_imputed[[as.character(sid)]] <- completed
}

# -------- Save Combined Data --------
combined_df <- bind_rows(all_imputed)
write.csv(combined_df, output_file, row.names = FALSE)
cat("✅ Saved combined imputed dataset to:", output_file, "\n")
