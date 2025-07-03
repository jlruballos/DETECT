# -------------------------------
# Global MICE Imputation + Diagnostics (No One-Hot Encoding)
# -------------------------------

library(mice)
library(ggplot2)
library(dplyr)
library(tidyr)
library(data.table)
library(zoo)     # for rollapply
library(purrr)

# -------- CONFIG --------
method_name <- "pmm"  # You can change this to "cart", "mean", etc.

# -------- Load Data --------
df <- read.csv("D:/DETECT/OUTPUT/raw_export_for_r/raw_daily_data_all_subjects.csv")
df$date <- as.Date(df$date)

# -------- Feature Categories --------
CLINICAL_FEATURES <- c("amyloid", "alzdis")
DEMO_FEATURES <- c("birthyr", "sex", "hispanic", "race", "educ", "livsitua", "independ", "residenc")

exclude_cols <- c("subid", "date", "label_fall", "label_hospital", "label", 
                  "days_since_fall", "days_until_fall", "days_since_hospital", "days_until_hospital")

# Define base features (excluding outcome/ID vars and demo/clinical)
base_feature_cols <- setdiff(colnames(df), c(exclude_cols, CLINICAL_FEATURES, DEMO_FEATURES))
feature_cols <- c(base_feature_cols, CLINICAL_FEATURES, DEMO_FEATURES)

# -------- Add Lag Features (only for time-varying features) --------
lag_days <- c(1, 7)
laggable_feature_cols <- base_feature_cols

df <- df %>%
  arrange(subid, date) %>%
  group_by(subid) %>%
  mutate(across(all_of(laggable_feature_cols),
                .fns = setNames(lapply(lag_days, function(l) function(x) lag(x, l)),
                                paste0("lag", lag_days)),
                .names = "{.col}_{.fn}")) %>%
  ungroup()

# -------- Update Feature Columns to Include Lags --------
lagged_feature_cols <- grep("_lag\\d+$", colnames(df), value = TRUE)
mice_feature_cols <- c(feature_cols, lagged_feature_cols)

# -------- Store Missingness Locations --------
missing_mask <- as.data.frame(is.na(df[, feature_cols]))

# -------- Run Global MICE Imputation --------
imp <- mice(df[, mice_feature_cols], m = 1, method = method_name, seed = 42)
completed_df <- complete(imp, 1)

# -------- Add Back Metadata --------
completed_df$subid <- df$subid
completed_df$date <- df$date
completed_df$label_fall <- df$label_fall
completed_df$label_hospital <- df$label_hospital
completed_df$label <- df$label
completed_df$days_since_fall <- df$days_since_fall
completed_df$days_until_fall <- df$days_until_fall
completed_df$days_since_hospital <- df$days_since_hospital
completed_df$days_until_hospital <- df$days_until_hospital

# -------- Add Imputed Flags --------
for (feature in feature_cols){
  completed_df[[paste0(feature, "_imputed")]] <- as.integer(missing_mask[[feature]])
}

# -------- Imputation Percent Per Subject --------
imputed_pct_per_subject <- completed_df %>%
  group_by(subid) %>%
  summarise(across(ends_with("_imputed"),
                   ~ round(100 * sum(.x, na.rm = TRUE)/n(), 1),
                   .names = "{.col}_pct"))

completed_df <- left_join(completed_df, imputed_pct_per_subject, by = "subid")

# -------- Temporal Feature Engineering (only for base features) --------
for (col in base_feature_cols) {
  completed_df[[paste0(col, "_delta")]] <- NA_real_
  completed_df[[paste0(col, "_delta_1d")]] <- NA_real_
  completed_df[[paste0(col, "_norm")]] <- NA_real_
  completed_df[[paste0(col, "_ma_7")]] <- NA_real_
}

for (sid in unique(completed_df$subid)) {
  sub_df <- completed_df[completed_df$subid == sid, ]
  sub_df <- sub_df[order(sub_df$date), ]
  
  for (col in base_feature_cols) {
    vals <- sub_df[[col]]
    mean_val <- mean(vals, na.rm = TRUE)
    sd_val <- sd(vals, na.rm = TRUE)
    delta <- vals - mean_val
    delta_1d <- c(NA, diff(vals))
    norm <- ifelse(sd_val != 0, delta / sd_val, NA)
    ma_7 <- zoo::rollapply(vals, width = 7, FUN = mean, fill = NA, align = "right", partial = TRUE)
    
    completed_df[completed_df$subid == sid, paste0(col, "_delta")] <- delta
    completed_df[completed_df$subid == sid, paste0(col, "_delta_1d")] <- delta_1d
    completed_df[completed_df$subid == sid, paste0(col, "_norm")] <- norm
    completed_df[completed_df$subid == sid, paste0(col, "_ma_7")] <- ma_7
  }
}

# -------- Save Output --------
write.csv(completed_df,
          paste0("D:/DETECT/OUTPUT/raw_export_for_r/labeled_daily_data_", method_name, ".csv"),
          row.names = FALSE)

# -------- Diagnostic Plots --------
output_root <- file.path("D:/DETECT/OUTPUT/raw_export_for_r/diagnostics", method_name)
dir.create(output_root, showWarnings = FALSE, recursive = TRUE)

unique_subs <- unique(df$subid)

for (s in unique_subs) {
  original_sub <- df %>% filter(subid == s)
  imputed_sub <- completed_df %>% filter(subid == s)
  mask_sub <- missing_mask[df$subid == s, , drop = FALSE]
  
  sub_dir <- file.path(output_root, paste0("subid_", s))
  histo_dir <- file.path(sub_dir, "histograms")
  line_dir <- file.path(sub_dir, "lineplots")
  dir.create(sub_dir, showWarnings = FALSE, recursive = TRUE)
  dir.create(histo_dir, showWarnings = FALSE)
  dir.create(line_dir, showWarnings = FALSE)
  
  for (col in feature_cols) {
    orig_vals <- original_sub[[col]]
    imputed_vals <- imputed_sub[[col]]
    
    if (all(is.na(orig_vals)) || all(is.na(imputed_vals))) next
    
    # ----- Overlaid Histogram -----
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
    
    # ----- Time Series with Red Dots + % Imputed -----
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
  }
}

