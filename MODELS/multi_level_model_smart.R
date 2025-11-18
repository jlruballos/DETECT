library(lme4)
library(lmerTest)
library(dplyr)
library(ggplot2)
library(tidyverse) #includes dplyr, ggplot2, readr, etc. 
library(forcats) #factor manipulation

# Load dataset
df <- read_csv("D:/DETECT/OUTPUT/SMART_merge_demographics_yearly/smart_with_yearly_demographics.csv")

#checks
str(df)
head(df)
summary(df$total_test_time)

# Factorize key/categorical variables
#facotr()converts numeric codes (1/2) into categorical data so R treats them as labels, not numeric magnitudes.
#cox models require categorical variables as factors so dummy variables are created
df$sex <- factor(df$sex)
df$age_ <- factor(df$age_bucket)
df$livsitua_recoded <- factor(df$livsitua_recoded)
df$educ_group <- factor(df$educ_group)
df$moca_ <- factor(df$moca_category)
df$maristat_ <- factor(df$maristat_recoded)
df$race_group <- factor(df$race_group)

#Recode numeric factor levels into readable labels for visualization
df$sex <- fct_recode(df$sex,
                     "_Male" = "1",
                     "_Female" = "2"
)

#Create test_index per subject

df <- df %>%
  arrange(subid, date) %>% #order by subid then time
  group_by(subid) %>%
  mutate(test_index = row_number()) %>%
  ungroup()

#check
df %>%
  group_by(subid) %>%
  summarise(
    n_tests = n(),
    min_index = min(test_index),
    max_index = max(test_index)
  ) %>%
  head()

df_model <- df %>%
  filter(
    !is.na(total_test_time),
    total_test_time > 0
  ) %>%
  mutate(
    time = as.numeric(total_test_time),
    log_time = log(time)
  ) %>%
  filter(is.finite(log_time))


#check
hist(df_model$total_test_time,  main = "Raw time", xlab = "time (sec)")
hist(df_model$log_time, main = "Log(time)", xlab = "log time")

#mixed model
model_smart <- lmer(
  log_time ~ 1 + (age_ + sex + educ_group) * test_index +
    (1 + test_index | subid),
  data = df_model,
  control = lmerControl(
    optimizer = "bobyqa",
    optCtrl = list(maxfun = 1e5)   # allow more iterations
  )
)

print("Summar Model")
print(summary(model_smart))

# Predictions
df_model$pred_log_time <- predict(model_smart)
df_model$pred_time <- exp(df_model$pred_log_time)

# ---- NEW: per-participant PNGs ----
plot_dir <- "D:/DETECT/OUTPUT/SMART_models/participant_plots"
dir.create(plot_dir, recursive = TRUE, showWarnings = FALSE)

participant_ids <- sort(unique(df_model$subid))

for (sid in participant_ids) {
  df_sub <- df_model %>% 
    filter(subid == sid) %>% 
    arrange(test_index)
  
  p <- ggplot(df_sub, aes(x = test_index, y = time)) +
    geom_point(alpha = 0.6) +
    geom_line(aes(y = pred_time), color = "red") +
    labs(
      title = paste("SMART time: subid", sid),
      x = "Test index",
      y = "Time (seconds)"
    ) +
    theme_minimal()
  
  file_name <- file.path(plot_dir, paste0("SMART_subid_", sid, ".png"))
  ggsave(file_name, p, width = 6, height = 4, dpi = 300)
}

# Residual diagnostics
par(mfrow = c(1, 2))
plot(model_smart)            # residuals vs fitted
qqnorm(residuals(model_smart))
qqline(residuals(model_smart))
par(mfrow = c(1, 1))

dir.create("D:/DETECT/OUTPUT/SMART_models", recursive = TRUE, showWarnings = FALSE)

saveRDS(model_smart, "D:/DETECT/OUTPUT/SMART_models/model_smart_c.rds")

fe <- fixef(model_smart)
exp_fe <- exp(fe)
pct_change <- 100 * (exp_fe - 1)
data.frame(
  term = names(fe),
  estimate_log = fe,
  estimate_mult = exp_fe,
  pct_change = pct_change
)


