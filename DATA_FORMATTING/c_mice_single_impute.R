# -------------------------------
# IMPROVED MICE Imputation with Fixed Temporal Features
# -------------------------------

library(mice)
library(ggplot2)
library(dplyr)
library(tidyr)
library(data.table)
library(purrr)
library(zoo)

# -------- CONFIG --------
method_name <- "pmm"  # Change this to "cart", "mean", etc. when needed

# -------- Load Data --------
df <- read.csv("D:/DETECT/OUTPUT/raw_export_for_r/raw_daily_data_all_subjects.csv")
df$date <- as.Date(df$date)

#---------Select one subject for testing----------
target_subid <- "2680"
df <- df %>% filter(subid == target_subid)

# -------- Define Feature Columns --------
exclude_cols <- c("subid", "date", "label_fall", "label_hospital", "label", 
                  "days_since_fall", "days_until_fall", "days_since_hospital", 
                  "days_until_hospital")
feature_cols <- setdiff(colnames(df), exclude_cols)

# -------- Add Lag Features per Subject --------
lag_days <- c(1,7)

df <- df %>%
  arrange(subid, date) %>%
  group_by(subid) %>%
  mutate(across(all_of(feature_cols),
                .fns = setNames(lapply(lag_days, function(l) function(x) lag(x, l)),
                                paste0("lag", lag_days)),
                .names = "{.col}_{.fn}")) %>%
  ungroup()

# -------- Update Feature Columns to Include Lags --------
lagged_feature_cols <- grep("_lag\\d+$", colnames(df), value = TRUE)
mice_feature_cols <- c(feature_cols, lagged_feature_cols)

# -------- Store Missingness Locations --------
missing_mask <- df %>% select(all_of(feature_cols)) %>% is.na() %>% as.data.frame()

# -------- Impute per Subject using MICE --------
impute_subject <- function(sub_df) {
  cat("Starting MICE imputation...\n")
  
  # Check if we have enough data for imputation
  if(nrow(sub_df) < 3) {
    warning("Not enough rows for reliable imputation")
    return(sub_df)
  }
  
  imp <- mice(sub_df[, mice_feature_cols], m = 1, method = method_name, 
              seed = 42, printFlag = FALSE)
  completed <- complete(imp, 1)
  
  # Reattach non-feature metadata in correct order
  completed$subid <- sub_df$subid
  completed$date <- sub_df$date
  completed$label_fall <- sub_df$label_fall
  completed$label_hospital <- sub_df$label_hospital
  completed$label <- sub_df$label
  completed$days_since_fall <- sub_df$days_since_fall
  completed$days_until_fall <- sub_df$days_until_fall
  completed$days_since_hospital <- sub_df$days_since_hospital
  completed$days_until_hospital <- sub_df$days_until_hospital
  
  cat("MICE imputation completed successfully!\n")
  return(completed)
}

completed_df <- impute_subject(df)

# Track which values were originally missing
for (feature in feature_cols){
  completed_df[[paste0(feature, "_imputed")]] <- as.integer(missing_mask[[feature]])
}

# -------- Add imputed flags and percent imputed --------
# Calculate percent of imputed values per subject for each feature
imputed_pct_per_subject <- completed_df %>%
  group_by(subid) %>%
  summarise(across(ends_with("_imputed"),
                   ~ round(100 * sum(.x, na.rm = TRUE)/n(), 1),
                   .names = "{.col}_pct"))

# Merge those percent values back into the main dataset
completed_df <- left_join(completed_df, imputed_pct_per_subject, by = "subid")

# -------- FIXED: Temporal Feature Engineering --------
cat("Creating temporal features...\n")

# APPROACH 1: Vectorized operations using dplyr (MUCH faster and cleaner)
completed_df <- completed_df %>%
  arrange(subid, date) %>%  # Ensure proper temporal ordering
  group_by(subid) %>%
  mutate(
    # Create all temporal features at once using across()
    across(all_of(feature_cols), 
           list(
             # Delta from subject mean
             delta = ~ .x - mean(.x, na.rm = TRUE),
             
             # Day-to-day change (first difference)
             delta_1d = ~ c(NA, diff(.x)),
             
             # Z-score normalization
             norm = ~ {
               subj_mean <- mean(.x, na.rm = TRUE)
               subj_sd <- sd(.x, na.rm = TRUE)
               if(subj_sd > 0 & !is.na(subj_sd)) {
                 (.x - subj_mean) / subj_sd
               } else {
                 rep(NA_real_, length(.x))
               }
             },
             
             # 7-day moving average (handles edge cases better)
             ma_7 = ~ {
               if(length(.x) >= 7) {
                 zoo::rollapply(.x, width = 7, FUN = mean, na.rm = TRUE, 
                                fill = NA, align = "right", partial = TRUE)
               } else {
                 # For short series, use expanding window
                 zoo::rollapply(.x, width = seq_along(.x), FUN = mean, na.rm = TRUE,
                                fill = NA, align = "right", partial = TRUE)
               }
             }
           ),
           .names = "{.col}_{.fn}"
    )
  ) %>%
  ungroup()

cat("Temporal features created successfully!\n")

# -------- Validation: Check our temporal features --------
cat("Validating temporal features...\n")

# Check for any issues with our new features
temporal_features <- grep("_(delta|delta_1d|norm|ma_7)$", colnames(completed_df), value = TRUE)

validation_summary <- completed_df %>%
  select(all_of(temporal_features)) %>%
  summarise(across(everything(), 
                   list(
                     n_missing = ~ sum(is.na(.x)),
                     n_infinite = ~ sum(is.infinite(.x)),
                     n_total = ~ length(.x)
                   )))

print("Temporal feature validation:")
print(validation_summary)

# -------- Save Imputed Data --------
output_file <- paste0("D:/DETECT/OUTPUT/raw_export_for_r/labeled_daily_data_", 
                      method_name, "_", target_subid, ".csv")
write.csv(completed_df, output_file, row.names = FALSE)
cat("Data saved to:", output_file, "\n")

# -------- Create Diagnostic Plots --------
output_root <- file.path("D:/DETECT/OUTPUT/raw_export_for_r/diagnostics", 
                         method_name, paste0("subid", target_subid))
dir.create(output_root, showWarnings = FALSE, recursive = TRUE)

s <- target_subid
original_sub <- df 
imputed_sub <- completed_df 
mask_sub <- missing_mask

sub_dir <- file.path(output_root, paste0("subid_", s))
histo_dir <- file.path(sub_dir, "histograms")
line_dir <- file.path(sub_dir, "lineplots")
dir.create(sub_dir, showWarnings = FALSE, recursive = TRUE)
dir.create(histo_dir, showWarnings = FALSE)
dir.create(line_dir, showWarnings = FALSE)

cat("Creating diagnostic plots...\n")

for (col in feature_cols) {
  # Skip if column doesn't exist or is all missing
  if (!col %in% colnames(original_sub) || !col %in% colnames(imputed_sub)) {
    warning(paste("Column", col, "not found, skipping..."))
    next
  }
  
  orig_vals <- original_sub[[col]]
  imputed_vals <- imputed_sub[[col]]
  
  if (all(is.na(orig_vals)) || all(is.na(imputed_vals))) {
    cat("Skipping", col, "- all values are missing\n")
    next
  }
  
  # ----- Overlaid Histogram -----
  tryCatch({
    overlay_df <- data.frame(
      value = c(orig_vals[!is.na(orig_vals)], imputed_vals[mask_sub[, col]]),
      type = c(rep("Original", sum(!is.na(orig_vals))),
               rep("Imputed", sum(mask_sub[, col], na.rm = TRUE)))
    )
    
    g_hist <- ggplot(overlay_df, aes(x = value, fill = type)) +
      geom_histogram(alpha = 0.5, position = "identity", bins = 30) +
      scale_fill_manual(values = c("Original" = "gray", "Imputed" = "red")) +
      labs(title = paste("Histogram:", col, "- Subject", s),
           x = col, y = "Count") +
      theme_minimal()
    
    ggsave(filename = file.path(histo_dir, paste0(col, "_hist_overlay_", s, ".png")),
           plot = g_hist, width = 8, height = 4)
  }, error = function(e) {
    warning(paste("Error creating histogram for", col, ":", e$message))
  })
  
  # ----- Time Series with Red Dots + % Imputed -----
  tryCatch({
    if (length(imputed_vals) < 3 || all(is.na(imputed_vals))) next
    
    imputed_flag <- mask_sub[, col]
    n_imputed <- sum(imputed_flag, na.rm = TRUE)
    total_vals <- sum(!is.na(orig_vals)) + n_imputed
    imputed_pct <- round(100 * n_imputed / total_vals, 1)
    
    plot_df <- imputed_sub
    plot_df$imputed <- imputed_flag
    
    g_line <- ggplot(plot_df, aes(x = date, y = !!sym(col))) +
      geom_line(color = "black") +
      geom_point(data = subset(plot_df, imputed == TRUE),
                 aes(x = date, y = !!sym(col)),
                 color = "red", size = 2) +
      labs(title = paste("Imputation for", col, "- Subject", s),
           subtitle = paste(n_imputed, "imputed points (", imputed_pct, "%)", sep = ""),
           y = col, x = "Date") +
      theme_minimal()
    
    ggsave(filename = file.path(line_dir, paste0(col, "_", s, ".png")),
           plot = g_line, width = 8, height = 4)
  }, error = function(e) {
    warning(paste("Error creating line plot for", col, ":", e$message))
  })
}

cat("Script completed successfully!\n")