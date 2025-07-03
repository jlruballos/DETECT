# ---------------------------------------
# R Script: Recurrent Event Modeling with reReg
# Based on interval-format export from Python
# ---------------------------------------

# Load libraries
library(tidyverse)
library(reReg)

# Load your interval-based dataset
df <- read_csv("D:/DETECT/OUTPUT/raw_export_for_r/intervals.csv")

# Quick check
glimpse(df)

# 🧹 Optional: factorize or clean covariates
df$sex <- factor(df$sex)
df$treat <- factor(df$amyloid, labels = c("negative", "positive"))
df$hispanic <- factor(df$hispanic)
#df$age <- factor(df$age)
df$age_cat <- cut(df$age, breaks = c(65, 70, 75, 80, 85, 100), right = FALSE)
df$age_cat <- factor(df$age_cat)
df$race <- factor(df$race)
df$livsitua <- factor(df$livsitua)
df$educ <- factor(df$educ)
df$residenc <- factor(df$residenc)
df$independ <- factor(df$independ)

#replace alzdis vlanks with 0s
df$alzdis[is.na(df$alzdis)] <- 0

cat("Event count (status):\n")
print(table(df$status))

cat("Unique values per demographic covariate:\n")
print(sapply(df[, c("hispanic", "race", "educ", "livsitua", "independ", "residenc", "alzdis", "age_cat")], function(x) length(unique(x))))

#summary(mod_base)

# Identify missing values in columns used
covars <- c("age_cat", "sex", "treat", "hispanic", "race", "educ", "livsitua", "residenc", 
            "alzdis","gait_speed_delta", 
            "steps_delta", "avghr_delta", 
            "durationinsleep_delta", "tossnturncount_delta")
print(colSums(is.na(df[, covars])))

#drop NAs
df <- df %>% drop_na(all_of(covars))

# Create Recur() object (like Surv(), but for recurrent events)
# Note: tstart and tstop are already in "days since start"
recur_obj <- Recur(time = df$tstop, id = df$id, event = df$status, origin = 0)

# Fit a baseline model (no predictors)
mod_base <- reReg::reReg(recur_obj ~ 1, data = df, model = "cox")

#Fit model with covariates (age, sex, amyloid)
mod_cov <- reReg::reReg(
                 recur_obj ~ age_cat + sex + treat + hispanic + race + educ + livsitua + residenc + alzdis + gait_speed_delta + steps_delta +
                   avghr_delta + durationinsleep_delta + tossnturncount_delta,
                 data = df,
                 model = "cox",
                 B = 200,
                 se = "boot"
                
                 )
summary(mod_cov)

mod_cov$par1  # Coefficients
mod_cov$par1.se  # Standard errors
mod_cov$par1.vcov  # Variance-covariance matrix

print(mod_cov$par1)         # Coefficients (log hazard ratios)
print(mod_cov$par1.se)      # Standard errors
print(exp(mod_cov$par1))    # Hazard ratios

# Plot cumulative mean function (CMF)
#plotCSM(mod_cov, main = "Cumulative Mean Function (CMF)", xlab = "Days", ylab = "Expected # of Falls")

# View individual subject trajectories (optional)
print(
  plotEvents(recur_obj,
             xlab = "Number of Days in Study",
             recurrent.name = "Falls"
             ) #, group = df$id)
)
# Predict mean number of events over time
#reReg::plotCSM(mod_cov)

# Extract estimated cumulative mean function
print(
  plot(mod_cov, 
       main = "Cumulative Mean Function", 
       xlab = "Days", 
       ylab = "Expected Number of Falls")
)

print(summary(mod_cov))

library(corrplot)
corr_data <- df %>%
  select(age , educ ,  gait_speed_delta, steps_delta, end_sleep_time_delta, 
         durationinsleep_delta, durationawake_delta, waso_delta, time_to_sleep_delta, 
         time_in_bed_after_sleep_delta, total_time_in_bed_delta, tossnturncount_delta, 
         sleep_period_delta, minhr_delta, maxhr_delta, avghr_delta, avgrr_delta, maxrr_delta, minrr_delta) %>%
  cor(use = "complete.obs")

corrplot(corr_data, method = "color", addCoef.col = "black", tl.col = "red", tl.cex = 1.2, number.cex = 0.8)


