# ============================================================
# Cox Mixed-Effects (Frailty) Comparison: Mean vs Delta vs Rolling
# ============================================================

# ---- Libraries ----
library(tidyverse)
library(survival)
library(coxme)
library(ggplot2)
library(forcats)
library(stringr)

# ---- Load Data ----
df <- read_csv("D:/DETECT/OUTPUT/survival_intervals/intervals_all_participants_label_fall.csv")
event_label <- "Fall"

# ---- Factorize categorical variables ----
df$sex        <- factor(df$sex)
df$amyloid    <- factor(df$amyloid)
df$hispanic   <- factor(df$hispanic)
df$age_       <- factor(df$age_bucket)
df$livsitua   <- factor(df$livsitua)
df$educ_group <- factor(df$educ_group)
df$moca_      <- factor(df$moca_category)
df$residenc   <- factor(df$residenc)
df$maristat_  <- factor(df$maristat_recoded)
df$cogstat    <- factor(df$cogstat)
df$race_group <- factor(df$race_group)

# ---- Recode for readability (optional) ----
df$sex <- fct_recode(df$sex, "_Male" = "1", "_Female" = "2")
df$amyloid <- fct_recode(df$amyloid, "_Negative" = "0", "_Positive" = "1")

# ---- Baseline covariates ----
baseline_covars <- c("sex", "amyloid", "moca_", "livsitua", "residenc",
                     "maristat_", "cogstat")

# ---- Event summary ----
df_frail <- df %>%
  select(id, tstart, tstop, status) %>%
  drop_na() %>%
  filter(tstop > tstart)

event_summary <- df_frail %>%
  group_by(id) %>%
  summarise(events = sum(status)) %>%
  summarise(
    mean_events = mean(events),
    median_events = median(events),
    max_events = max(events)
  )
print(event_summary)

# ============================================================
# Function to run CoxME frailty model safely
# ============================================================

run_coxme_model <- function(df, feature_suffix, label) {
  sensor_vars <- names(df)[grepl(feature_suffix, names(df))]
  covars <- c(baseline_covars, sensor_vars)
  
  temp_df <- df %>%
    drop_na(all_of(covars), tstart, tstop, status, id) %>%
    filter(tstop > tstart) %>%
    droplevels()
  
  cat("\n============================\n")
  cat("Running model:", label, "\n")
  cat("============================\n")
  cat("Rows used:", nrow(temp_df), " | Covariates:", length(covars), "\n")
  
  if (nrow(temp_df) < 100) {
    cat("⚠️ Skipping — not enough data.\n")
    return(NULL)
  }
  
  model_formula <- as.formula(
    paste("Surv(tstart, tstop, status) ~",
          paste(covars, collapse = " + "),
          "+ (1 | id)")
  )
  
  model <- tryCatch({
    coxme(model_formula, data = temp_df)
  }, error = function(e) {
    cat("❌ Model failed:", e$message, "\n")
    return(NULL)
  })
  
  if (!is.null(model)) {
    print(summary(model))
  }
  return(model)
}

# ============================================================
# Run models
# ============================================================

mod_mean  <- run_coxme_model(df, "_interval_mean",  "Interval Mean Features")
mod_roll  <- run_coxme_model(df, "_ma7_last",       "Rolling Mean (7-day) Features")
mod_delta <- run_coxme_model(df, "_delta",          "Delta Features")

# ============================================================
# Extract summaries
# ============================================================

extract_summary <- function(model, model_name) {
  if (is.null(model)) return(NULL)
  
  coefs <- coef(model)
  if (length(coefs) == 0) return(NULL)
  
  tibble(
    variable = names(coefs),
    estimate = as.numeric(coefs),
    std.error = sqrt(diag(vcov(model))),
    hazard_ratio = exp(as.numeric(coefs)),
    model = model_name
  )
}

summary_list <- list(
  extract_summary(mod_mean, "Interval Mean"),
  extract_summary(mod_roll, "Rolling Mean"),
  extract_summary(mod_delta, "Delta")
)
summary_list <- summary_list[!sapply(summary_list, is.null)]
all_summaries <- bind_rows(summary_list)

# ============================================================
# Output CSV + Print top results
# ============================================================

output_path <- paste0("D:/DETECT/OUTPUT/R_Output/coxme_model_summary_", event_label, ".csv")
dir.create(dirname(output_path), showWarnings = FALSE, recursive = TRUE)
write_csv(all_summaries, output_path)
cat("✅ CSV written to:", output_path, "\n")

print(all_summaries %>% arrange(model, desc(hazard_ratio)) %>% head(10))

# ============================================================
# Plot Hazard Ratios
# ============================================================

if (nrow(all_summaries) > 0) {
  all_summaries <- all_summaries %>%
    mutate(
      lower = exp(estimate - 1.96 * std.error),
      upper = exp(estimate + 1.96 * std.error)
    )
  
  hr_plot <- ggplot(all_summaries, aes(x = reorder(variable, hazard_ratio),
                                       y = hazard_ratio, color = model)) +
    geom_point(size = 3) +
    geom_errorbar(aes(ymin = lower, ymax = upper), width = 0.2) +
    coord_flip() +
    geom_hline(yintercept = 1, linetype = "dashed", color = "gray50") +
    labs(title = "Hazard Ratios (Cox Mixed-Effects Models)",
         y = "Hazard Ratio", x = "") +
    theme_minimal(base_size = 14)
  
  print(hr_plot)
  
  ggsave("D:/DETECT/OUTPUT/R_Output/Graphs/coxme_hazard_ratios.png",
         plot = hr_plot, width = 10, height = 7, dpi = 300)
}

# ============================================================
# Frailty Extraction and Visualization (from Mean Model)
# ============================================================

if (!is.null(mod_mean)) {
  frailty_scores <- data.frame(
    id = names(ranef(mod_mean)$id),
    frailty = as.numeric(ranef(mod_mean)$id)
  ) %>%
    mutate(risk_multiplier = exp(frailty))
  
  print(head(frailty_scores))
  
  frailty_plot1 <- ggplot(frailty_scores, aes(x = frailty)) +
    geom_histogram(bins = 20, fill = "steelblue", color = "white") +
    labs(title = "Distribution of Subject Frailty (log scale)",
         x = "log-frailty", y = "Count") +
    theme_minimal(base_size = 14)
  
  frailty_plot2 <- ggplot(frailty_scores,
                          aes(x = reorder(id, frailty), y = frailty)) +
    geom_point(color = "darkred", size = 2.5) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "gray50") +
    coord_flip() +
    labs(title = "Participant-specific Frailty Estimates",
         x = "Participant ID", y = "log-frailty") +
    theme_minimal(base_size = 14)
  
  print(frailty_plot1)
  print(frailty_plot2)
  
  # Save frailty data
  write_csv(frailty_scores,
            "D:/DETECT/OUTPUT/R_Output/frailty_scores_coxme.csv")
  cat("✅ Frailty scores exported.\n")
  
  # Variance summary
  frailty_var <- as.numeric(VarCorr(mod_mean))
  cat("\nEstimated Frailty Variance:", round(frailty_var, 3),
      " | SD =", round(sqrt(frailty_var), 3), "\n")
  cat("Hazard multiplier for +1 SD frailty:",
      round(exp(sqrt(frailty_var)), 2), "x\n")
}

cat("\n✅ All models completed successfully.\n")
