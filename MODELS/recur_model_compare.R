# ---------------------------------------
# Full R Script: Compare Mean vs Delta vs Rolling Mean Models
# ---------------------------------------

# Load libraries
library(tidyverse) #includes dplyr, ggplot2, readr, etc. 
library(reReg) #recurrent event regression modeling
library(reda) #tool for recurrent event data (plotting etc.)

library(dplyr) #data manipulation
library(forcats) #factor manipulation
library(stringr) #string manipulation

# Load dataset
df <- read_csv("D:/DETECT/OUTPUT/survival_intervals/intervals_all_participants_label_fall.csv")

event_label <- "Fall"  # Change this to match the current event type

# Factorize key/categorical variables
#cox models require categorical variables as factors so dummy variables are created
df$sex <- factor(df$sex)
#df$amyloid <- factor(df$amyloid, labels = c("negative", "positive"))
df$amyloid <- factor(df$amyloid)
df$hispanic <- factor(df$hispanic)
#df$age <- factor(df$age)
#df$age_cat <- cut(df$age, breaks = c(65, 70, 75, 80, 85, 100), right = FALSE)
#df$age_cat <- factor(df$age_cat)
df$age_ <- factor(df$age_bucket)
#df$age_ <- factor(df$age_at_visit)
#df$race <- factor(df$race)
# Create a new column 'race_binary' to store the recoded values
#df$race_binary <- ifelse(df$race == "White", 1, 0) #race white is 1 and everything else is 0 or non-white
df$livsitua <- factor(df$livsitua)
df$livsitua_binary <- ifelse(df$livsitua == "lives_alone", 1, 0) #lives alone is 1 everything else is 0 or does not live alone
#df$educ <- factor(df$educ)
df$educ_group <- factor(df$educ_group)
df$moca_ <- factor(df$moca_category)
df$residenc <- factor(df$residenc)
#df$maristat <- factor(df$maristat)
df$maristat_ <- factor(df$maristat_recoded)
df$cogstat <- factor(df$cogstat)
df$race_group <- factor(df$race_group)
#df$moca_cat <- cut(df$moca_avg, breaks = c(18, 26, 30), right = FALSE)
#df$moca_avg <- factor(df$moca_avg)

# Replace NA in alzdis with 0
#df$alzdis[is.na(df$alzdis)] <- 0

#df$alzdis <- factor(df$alzdis)

#df$group <- paste0("Educ ", df$educ, " | Age ", df$age_cat, " | ", df$amyloid)

df$group <- paste0( " | ", df$amyloid)

#recoding block

# df <- df %>%
#   rename(
#     maristat = maristat_recoded,
#     age = age_at_visit,
#     moca = moca_category
#   )

#recoding with fct_recode to give readable labels for plots.

df$sex <- fct_recode(df$sex,
                     "_Male" = "1",
                     "_Female" = "2"
)

df$amyloid <- fct_recode(df$amyloid,
                         "_Negative" = "0",
                         "_Positive" = "1"
)

df$livsitua <- fct_recode(df$livsitua,
                          "_Lives Alone" = "1",
                          "_Lives with Partner" = "2",
                          "_Lives with Relative" = "3",
                          "_Lives with Caregiver" = "4",
                          "_Group Residence" = "5",
                          "_Assisted Living" = "6",
                          "_Unknown" = "9"
)

# df$alzdis <- fct_recode(df$alzdis,
#                         "_Not Present" = "0",
#                         "_Present" = "1"
# )

df$cogstat <- fct_recode(df$cogstat,
                         "_Undeterminable" = "0",
                         "_Better than Normal" = "1",
                         "_Normal" = "2",
                         "_1–2 Scores Abnormal" = "3",
                         "_3+ Scores Abnormal" = "4"
)

df$residenc <- fct_recode(df$residenc,
                          "_Private Residence" = "1",
                          "_Retirement Community" = "2",
                          "_Assisted Living" = "3",
                          "_Skilled Nursing" = "4",
                          "_Unknown" = "9"
)

# Define baseline demographic covariates
#these demographic variables will always be included
baseline_covars <- c( "sex", "amyloid", "moca_",
                     "livsitua", "residenc",  "maristat_", "cogstat")

# baseline_covars <- c("age_at_visit", "sex", "amyloid",  "moca_category",
#                      "livsitua", "residenc", "maristat", "cogstat")

# Prepare data for plotting recurrent events (filter missing)
plot_df <- df %>% drop_na(tstart, tstop, status, id, group)

# Create Recur object for plotting
#each participant(id) can have multiple rows each with a time interval up to the next event
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
  
  # NEW: Only drop rows missing critical survival variables
  #critical_vars <- c("tstart", "tstop", "status", "id")
  #temp_df <- df %>% drop_na(all_of(critical_vars))
  
  cat("  → Rows used in model:", nrow(temp_df), "\n")
  
  # # Drop factor columns with only 1 level
  factor_covars <- covars[sapply(temp_df[covars], is.factor)]
  single_level_factors <- factor_covars[sapply(temp_df[factor_covars], function(x) length(unique(x)) <= 1)]

  if (length(single_level_factors) > 0) {
    message("Dropping variables with only 1 level: ", paste(single_level_factors, collapse = ", "))
    covars <- setdiff(covars, single_level_factors)
  }
  
  # Build Recur object
  recur_obj <- Recur(time = temp_df$tstop, id = temp_df$id, event = temp_df$status)
  
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
#extract_summary <- function(model, model_name) {
 # s <- summary(model)
  #tibble(
   # variable = rownames(s$coefficients),
    #estimate = s$coefficients[, "Estimate"],
    #std.error = s$coefficients[, "StdErr"],
  #  p_value = s$coefficients[, "p.value"],
   # model = model_name
  #)
#}

extract_summary <- function(model, model_name) {
  s <- tryCatch(summary(model), error = function(e) return(NULL))
  if (is.null(s)) return(NULL)
  
  if (!all(c("Estimate", "StdErr", "p.value") %in% colnames(s$coefficients))) {
    warning(paste("Model", model_name, "missing required columns."))
    return(NULL)
  }
  
  tibble(
    variable = rownames(s$coefficients),
    estimate = s$coefficients[, "Estimate"],
    std.error = s$coefficients[, "StdErr"],
    p_value = s$coefficients[, "p.value"],
    model = model_name
  )
}

# Extract summaries with correct model names
# summary_mean   <- extract_summary(mod_mean, "Interval Mean")
# summary_roll   <- extract_summary(mod_roll, "Rolling Mean")
# summary_delta  <- extract_summary(mod_delta, "Delta")

mod_counts     <- run_model(df, "_total", "Total Count Features")
summary_counts <- extract_summary(mod_counts, "Total Counts")

summary_list <- list(
  extract_summary(mod_mean, "Interval Mean"),
  extract_summary(mod_roll, "Rolling Mean"),
  extract_summary(mod_delta, "Delta") #,
 # extract_summary(mod_counts, "Total Counts")
)

# Filter out any NULL models that failed to extract
summary_list <- summary_list[!sapply(summary_list, is.null)]

all_summaries <- bind_rows(summary_list)


# --- Combine all into one table ---
# all_summaries <- bind_rows(summary_mean, summary_roll, summary_delta, summary_counts)

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

all_summaries <- all_summaries %>%
  mutate(
    hazard_ratio = exp(estimate),
    lower = exp(estimate - 1.96 * std.error),
    upper = exp(estimate + 1.96 * std.error)
  ) %>%
  arrange(desc(hazard_ratio))

# Get top 10 terms with p ≤ 0.5 and highest HR
top_tmp <- all_summaries %>%
  group_by(model) %>%
  filter(p_value <= 0.05) %>%
  arrange(desc(hazard_ratio)) %>%
  slice(1:5) %>%
  filter(hazard_ratio > 1) %>%
  filter(model %in% c("Interval Mean", "Rolling Mean"))

bot_tmp <- all_summaries %>%
  group_by(model) %>%
  filter(p_value <= 0.05) %>%
  arrange(hazard_ratio) %>%
  slice(1:5) %>%
  filter(hazard_ratio < 1) %>%
  filter(model %in% c("Interval Mean", "Rolling Mean"))

top_terms <- rbind(top_tmp, bot_tmp)

top_terms <- top_terms %>%
  mutate(
    direction = ifelse(hazard_ratio > 1, "Increased Risk", "Decreased Risk")
  )

# Plot
forest_plot <- ggplot(top_terms, aes(
  x = hazard_ratio, 
  y = reorder(variable, hazard_ratio), 
  color = direction)) +
  geom_vline(xintercept = 1, color = "gray75") +
  geom_linerange(aes(xmin = lower, xmax = upper), linewidth = 1.5, alpha = 0.5) +
  geom_point(size = 4) +
  theme_minimal(base_size = 16) +
  scale_color_manual(
    values = c("Increased Risk" = "red3", "Decreased Risk" = "green4")
  ) +
  xlim(c(0, max(top_terms$upper, na.rm = TRUE) + 0.5)) +
  labs(
    title = str_wrap(paste("Hazard Ratios:", event_label, "Each Week"), width = 30),
    subtitle = "p ≤ 0.05",
    x = "Hazard Ratio Estimate (95% CI)",
    y = NULL,
    color = NULL
  ) +
  theme(
    plot.title = element_text(size = 26),
    plot.subtitle = element_text(size = 24),
    plot.caption = element_text(size = 20),
    axis.title.x = element_text(size = 23),
    axis.title.y = element_text(size = 23),
    axis.text.x = element_text(size = 21),
    axis.text.y = element_text(size = 21),
    legend.title = element_text(size = 22),
    legend.text = element_text(size = 20),
    strip.text.x = element_text(size = 22),
    strip.text.y = element_text(size = 22),
    legend.position = "bottom",
    text = element_text(size = 20)
  ) +
  facet_wrap(~model)

# Save the plot
ggsave(
  filename = paste0("D:/DETECT/OUTPUT/R_Output/Graphs/Hazard_Ratio_Mean_", event_label, ".png"),
  plot = forest_plot,
  width = 12,
  height = 8,
  dpi = 300
)

# Show the plot
print(forest_plot)

# Get top 10 terms with p ≤ 0.5 and highest HR
top_tmp <- all_summaries %>%
  group_by(model) %>%
  filter(p_value <= 0.05) %>%
  arrange(desc(hazard_ratio)) %>%
  slice(1:5) %>%
  filter(hazard_ratio > 1) %>%
  filter(model %in% c("Delta"))

bot_tmp <- all_summaries %>%
  group_by(model) %>%
  filter(p_value <= 0.05) %>%
  arrange(hazard_ratio) %>%
  slice(1:5) %>%
  filter(hazard_ratio < 1) %>%
  filter(model %in% c("Delta"))

top_terms <- rbind(top_tmp, bot_tmp)

top_terms <- top_terms %>%
  mutate(
    direction = ifelse(hazard_ratio > 1, "Increased Risk", "Decreased Risk")
  )

# Plot
forest_plot_2 <- ggplot(top_terms, aes(
  x = hazard_ratio, 
  y = reorder(variable, hazard_ratio), 
  color = direction)) +
  geom_vline(xintercept = 1, color = "gray75") +
  geom_linerange(aes(xmin = lower, xmax = upper), linewidth = 1.5, alpha = 0.5) +
  geom_point(size = 4) +
  theme_minimal(base_size = 16) +
  scale_color_manual(
    values = c("Increased Risk" = "red3", "Decreased Risk" = "green4")
  ) +
  xlim(c(0, max(top_terms$upper, na.rm = TRUE) + 0.5)) +
  labs(
    title = str_wrap(paste("Hazard Ratios:", event_label, "Each Week"), width = 30),
    subtitle = "p ≤ 0.05",
    x = "Hazard Ratio Estimate (95% CI)",
    y = NULL,
    color = NULL
  ) +
  theme(
    plot.title = element_text(size = 26),
    plot.subtitle = element_text(size = 24),
    plot.caption = element_text(size = 20),
    axis.title.x = element_text(size = 23),
    axis.title.y = element_text(size = 23),
    axis.text.x = element_text(size = 21),
    axis.text.y = element_text(size = 21),
    legend.title = element_text(size = 22),
    legend.text = element_text(size = 20),
    strip.text.x = element_text(size = 22),
    strip.text.y = element_text(size = 22),
    legend.position = "bottom",
    text = element_text(size = 20)
  ) +
  facet_wrap(~model)

# Save the plot
ggsave(
  filename = paste0("D:/DETECT/OUTPUT/R_Output/Graphs/Hazard_Ratio_Delta_", event_label, ".png"),
  plot = forest_plot_2,
  width = 9,
  height = 8,
  dpi = 300
)

# Show the plot
print(forest_plot_2)

