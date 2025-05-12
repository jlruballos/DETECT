# STL Imputation with Error Handling and Plotting
library(dplyr)
library(ggplot2)
library(zoo)

# -------- CONFIG --------
input_path <- "D:/DETECT/OUTPUT/raw_export_for_r/raw_daily_data_all_subjects.csv"
output_path <- "D:/DETECT/OUTPUT/raw_export_for_r/stl_imputed"
dir.create(output_path, showWarnings = FALSE, recursive = TRUE)

# -------- LOAD DATA --------
df <- read.csv(input_path)
df$date <- as.Date(df$date)

# -------- Define Features --------
exclude_cols <- c("subid", "date")
feature_cols <- setdiff(colnames(df), exclude_cols)

# -------- Initialize Logs --------
skipped_log <- data.frame(subid = character(), feature = character(), reason = character(), stringsAsFactors = FALSE)
impute_summary <- data.frame(subid = character(), feature = character(), imputed_count = integer(), total_count = integer(), percent_imputed = numeric(), stringsAsFactors = FALSE)

# -------- Impute with STL --------
all_subids <- unique(df$subid)
for (sub in all_subids) {
  sub_df <- df %>% filter(subid == sub) %>% arrange(date)
  imputed_sub_df <- sub_df
  
  sub_output_dir <- file.path(output_path, paste0("sub-", sub))
  dir.create(sub_output_dir, showWarnings = FALSE)
  
  for (feature in feature_cols) {
    series <- sub_df[[feature]]
    
    if (all(is.na(series))) {
      skipped_log <- rbind(skipped_log, data.frame(subid = sub, feature = feature, reason = "all NA"))
      next
    }
    if (sum(!is.na(series)) < 14) {
      skipped_log <- rbind(skipped_log, data.frame(subid = sub, feature = feature, reason = "too short"))
      next
    }
    
    approx_series <- na.approx(series, na.rm = FALSE)
    if (sum(!is.na(approx_series)) < 7) {
      skipped_log <- rbind(skipped_log, data.frame(subid = sub, feature = feature, reason = "still too sparse"))
      next
    }
    
    ts_series <- ts(approx_series, frequency = 7)
    tryCatch({
      stl_fit <- stl(ts_series, s.window = "periodic")
      imputed_vals <- stl_fit$time.series[, "trend"] + stl_fit$time.series[, "seasonal"]
      imputed_series <- series
      imputed_series[is.na(imputed_series)] <- imputed_vals[is.na(imputed_series)]
      imputed_sub_df[[feature]] <- imputed_series
      
      imputed_count <- sum(is.na(series))
      total_count <- length(series)
      percent_imputed <- round(100 * imputed_count / total_count, 2)
      impute_summary <- rbind(impute_summary, data.frame(
        subid = sub, feature = feature, 
        imputed_count = imputed_count, 
        total_count = total_count, 
        percent_imputed = percent_imputed
      ))
      
      plot_data <- data.frame(date = sub_df$date, original = series, imputed = imputed_series)
      g <- ggplot(plot_data, aes(x = date)) +
        geom_line(aes(y = imputed), color = "blue") +
        geom_point(aes(y = original), color = "black") +
        labs(title = paste("STL Imputation:", feature, "- Subject", sub, sprintf("(%d/%.0f%%)", imputed_count, percent_imputed)), y = feature)
      if (any(is.na(series))) {
        g <- g + geom_point(data = subset(plot_data, is.na(original)), aes(y = imputed), color = "red", size = 2)
      }
      ggsave(file.path(sub_output_dir, paste0("stl_", feature, "_impute_plot.png")), g, width = 8, height = 4)
      
    }, error = function(e) {
      skipped_log <<- rbind(skipped_log, data.frame(subid = sub, feature = feature, reason = e$message))
    })
  }
  write.csv(imputed_sub_df, file.path(sub_output_dir, paste0("sub_", sub, "_stl_imputed.csv")), row.names = FALSE)
}

# -------- Save Logs --------
write.csv(skipped_log, file.path(output_path, "stl_skipped_log.csv"), row.names = FALSE)
write.csv(impute_summary, file.path(output_path, "stl_impute_summary.csv"), row.names = FALSE)
