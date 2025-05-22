# ---------------------------------------
# R Script: Recurrent Event Modeling with reReg
# Based on interval-format export from Python
# ---------------------------------------

# 📦 Load libraries
library(tidyverse)
library(reReg)

# 📁 Load your interval-based dataset
df <- read_csv("D:/DETECT/OUTPUT/raw_export_for_r/intervals.csv")

# 👀 Quick check
glimpse(df)

# 🧹 Optional: factorize or clean covariates
df$sex <- factor(df$sex)
df$treat <- factor(df$amyloid, labels = c("negative", "positive"))

# 📌 Create Recur() object (like Surv(), but for recurrent events)
# Note: tstart and tstop are already in "days since start"
recur_obj <- Recur(time = df$tstop, id = df$id, event = df$status, origin = 0)

# 🧪 Fit a baseline model (no predictors)
mod_base <- reReg::reReg(recur_obj ~ 1, data = df, model = "cox")
#summary(mod_base)

# ➕ Fit model with covariates (age, sex, amyloid)
mod_cov <- reReg::reReg(
                 recur_obj ~ age + sex + treat, 
                 data = df, 
                 model = "cox"
                 )
#summary(mod_cov)

mod_cov$par1  # Coefficients
mod_cov$par1.se  # Standard errors
mod_cov$par1.vcov  # Variance-covariance matrix

print(mod_cov$par1)         # Coefficients (log hazard ratios)
print(mod_cov$par1.se)      # Standard errors
print(exp(mod_cov$par1))    # Hazard ratios

# 📈 Plot cumulative mean function (CMF)
#plotCSM(mod_cov, main = "Cumulative Mean Function (CMF)", xlab = "Days", ylab = "Expected # of Falls")

# 🧠 View individual subject trajectories (optional)
print(
  plotEvents(recur_obj,
             xlab = "Number of Days in Study",
             recurrent.name = "Falls"
             ) #, group = df$id)
)
# 📊 Predict mean number of events over time
#reReg::plotCSM(mod_cov)

# Extract estimated cumulative mean function
print(
  plot(mod_cov, 
       main = "Cumulative Mean Function", 
       xlab = "Days", 
       ylab = "Expected Number of Falls")
)



# ✅ Done!
