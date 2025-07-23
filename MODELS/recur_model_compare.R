# ---------------------------------------
# Full R Script: Compare Mean vs Delta vs Rolling Mean Models
# ---------------------------------------

# Load libraries
library(tidyverse)
library(reReg)
library(reda) 

# Load dataset
df <- read_csv("D:/DETECT/OUTPUT/raw_export_for_r/intervals_label_mood_blue.csv")

event_label <- "Blue_1k"  # Change this to match the current event type

# Factorize key variables
df$sex <- factor(df$sex)
#df$amyloid <- factor(df$amyloid, labels = c("negative", "positive"))
df$amyloid <- factor(df$amyloid)
df$hispanic <- factor(df$hispanic)
#df$age <- factor(df$age)
df$age_cat <- cut(df$age, breaks = c(65, 70, 75, 80, 85, 100), right = FALSE)
df$age_cat <- factor(df$age_cat)
df$race <- factor(df$race)
df$livsitua <- factor(df$livsitua)
df$educ <- factor(df$educ)
df$residenc <- factor(df$residenc)
df$maristat <- factor(df$maristat)
df$cogstat <- factor(df$cogstat)
df$moca_cat <- cut(df$moca_avg, breaks = c(18, 26, 30), right = FALSE)
#df$moca_avg <- factor(df$moca_avg)

# Replace NA in alzdis with 0
df$alzdis[is.na(df$alzdis)] <- 0

df$alzdis <- factor(df$alzdis)

df$group <- paste0("Educ ", df$educ, " | Age ", df$age_cat, " | ", df$amyloid)


# Define baseline demographic covariates
baseline_covars <- c("age_cat", "sex", "amyloid", "race", 
                     "livsitua", "residenc", "alzdis", "maristat", "cogstat")

# Prepare data for plotting recurrent events (filter missing)
plot_df <- df %>% drop_na(tstart, tstop, status, id, group)

# Create Recur object for plotting
recur_obj_plot <- Recur(time = plot_df$tstop, id = plot_df$id, event = plot_df$status, origin = 0)

plot_df <- plot_df %>% arrange(id, tstop)

# ---------------------------------------
# Function to run and summarize model
# ---------------------------------------
run_model <- function(df, feature_suffix, label) {
  # Select sensor features with the desired suffix
  sensor_vars <- names(df)[grepl(feature_suffix, names(df))]
  covars <- c(baseline_covars, sensor_vars)
  
  # Drop rows with NA in any relevant covariates
  temp_df <- df %>% drop_na(all_of(covars))
  
  cat("  → Rows used in model:", nrow(temp_df), "\n")
  
  # # Drop factor columns with only 1 level
  factor_covars <- covars[sapply(temp_df[covars], is.factor)]
  single_level_factors <- factor_covars[sapply(temp_df[factor_covars], function(x) length(unique(x)) <= 1)]

  if (length(single_level_factors) > 0) {
    message("Dropping variables with only 1 level: ", paste(single_level_factors, collapse = ", "))
    covars <- setdiff(covars, single_level_factors)
  }
  
  # Build Recur object
  recur_obj <- Recur(time = temp_df$tstop, id = temp_df$id, event = temp_df$status, origin = 0)
  
  # Build formula
  model_formula <- as.formula(paste("recur_obj ~", paste(covars, collapse = " + ")))
  
  # Fit model
  mod <- reReg::reReg(
    formula = model_formula,
    data = temp_df,
    model = "cox",
    B = 1000,
    se = "boot"
  )
  
  cat("\n============================\n")
  cat("Model:", label, "\n")
  cat("============================\n")
  print(summary(mod))
  return(mod)
}

# ---------------------------------------
# Run models
# ---------------------------------------

mod_mean  <- run_model(df, "_interval_mean",  "Interval Mean Features")
mod_roll  <- run_model(df, "_ma7_last",       "Rolling Mean (7-day) Features")
mod_delta <- run_model(df, "_delta",          "Delta Features")



# Define extraction function
extract_summary <- function(model, model_name) {
  s <- summary(model)
  tibble(
    variable = rownames(s$coefficients),
    estimate = s$coefficients[, "Estimate"],
    p_value = s$coefficients[, "p.value"],
    model = model_name
  )
}

# Extract summaries with correct model names
summary_mean   <- extract_summary(mod_mean, "Interval Mean")
summary_roll   <- extract_summary(mod_roll, "Rolling Mean")
summary_delta  <- extract_summary(mod_delta, "Delta")

mod_counts     <- run_model(df, "_total", "Total Count Features")
summary_counts <- extract_summary(mod_counts, "Total Counts")

# --- Combine all into one table ---
all_summaries <- bind_rows(summary_mean, summary_roll, summary_delta, summary_counts)

# Add hazard ratio column
all_summaries <- all_summaries %>%
  mutate(hazard_ratio = exp(estimate)) %>%
  arrange(model, p_value)

print(
  all_summaries %>% 
    filter(p_value < 0.1) %>% 
    arrange(model, p_value)
)

print(
  all_summaries %>%
    select(model, variable, hazard_ratio, p_value) %>%
    print(n = Inf)
  
)

output_path <- paste0("D:/DETECT/OUTPUT/R_Output/recurrent_model_summary_", event_label, ".csv")

dir.create(dirname(output_path), showWarnings = FALSE, recursive = TRUE)

write_csv(
  all_summaries %>%
    select(model, variable, hazard_ratio, p_value),
  output_path
)

cat("✅ CSV written to:", output_path, "\n")

library(ggplot2)

print(
  all_summaries %>%
  filter(p_value < 0.1) %>%
  ggplot(aes(x = reorder(variable, hazard_ratio), y = hazard_ratio, fill = model)) +
  geom_bar(stat = "identity", position = "dodge") +
  coord_flip() +
  labs(title = "Hazard Ratios (HR < 0.1)", y = "HR", x = "") +
  theme_minimal()
)


plot_df$group_simple <- interaction(plot_df$sex, plot_df$amyloid)
print(
  plotEvents(
    Recur(tstop, id, status) ~ amyloid,
    data = plot_df,
    recurrent.name = event_label,
    xlab = "Days Since Start",
    main = "Recurrent by Amyloid Status",
    col.recurrent = "dodgerblue",
    legend = TRUE
  )
)

# Plot baseline rate function for the mean model
print(plotRate(mod_mean))

# Plot cumulative sample mean function by amyloid status
# Cumulative sample mean plot by amyloid
print(
  plot(Recur(tstop, id, status) ~ amyloid, data = plot_df, CSM = TRUE)
)


