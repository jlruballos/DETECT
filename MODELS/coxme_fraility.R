# ============================================================
# Cox Mixed-Effects (Frailty) Model for Recurrent Fall Events
# ============================================================

# ---- Load Libraries ----
library(dplyr)
library(readr)
library(survival)
library(coxme)
library(ggplot2)

# ---- Load Dataset ----
df <- read_csv("D:/DETECT/OUTPUT/survival_intervals/intervals_all_participants_label_fall.csv")

# ---- Prepare Data ----
df_frail <- df %>%
  dplyr::select(id, tstart, tstop, status,
                steps_interval_mean, awakenings_interval_mean, amyloid) %>%
  drop_na() %>%
  filter(tstop > tstart)  # Remove invalid intervals

# ---- Check Event Distribution ----
event_summary <- df_frail %>%
  group_by(id) %>%
  summarise(events = sum(status)) %>%
  summarise(
    mean_events   = mean(events),
    median_events = median(events),
    max_events    = max(events)
  )

print("========== EVENT SUMMARY ==========")
print(event_summary)

# ---- Fit Cox Mixed-Effects Model (Random Intercept per Participant) ----
coxme_model <- coxme(
  Surv(tstart, tstop, status) ~
    steps_interval_mean + awakenings_interval_mean + amyloid + (1 | id),
  data = df_frail
)

# ---- Model Summary ----
print("========== MODEL SUMMARY ==========")
print(summary(coxme_model))

# ---- Extract Fixed Effects (Population-Level Effects) ----
print("========== FIXED EFFECTS ==========")
print(coef(coxme_model))

# ---- Extract Random Effects (Subject-Level Frailty) ----
frailty_scores <- data.frame(
  id = names(ranef(coxme_model)$id),
  frailty = as.numeric(ranef(coxme_model)$id)
) %>%
  mutate(risk_multiplier = exp(frailty))

print("========== SAMPLE FRAILTIES ==========")
print(head(frailty_scores))

# ---- Summary Statistics ----
print(summary(frailty_scores$frailty))

# ---- Visualize Frailty Distribution ----
p1 <- ggplot(frailty_scores, aes(x = frailty)) +
  geom_histogram(bins = 20, fill = "steelblue", color = "white") +
  labs(
    title = "Distribution of Subject Frailty (log scale)",
    x = "log-frailty (random effect)",
    y = "Count"
  ) +
  theme_minimal(base_size = 14)

# ---- Plot Ranked Frailties ----
p2 <- ggplot(frailty_scores, aes(x = reorder(id, frailty), y = frailty)) +
  geom_point(color = "darkred", size = 2.5) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "gray50") +
  coord_flip() +
  labs(
    title = "Participant-specific Frailty Estimates",
    x = "Participant ID",
    y = "log-frailty (random effect)"
  ) +
  theme_minimal(base_size = 14)

# ---- Display Plots ----
print(p1)
print(p2)

# ---- Export Frailty Scores ----
output_path <- "D:/DETECT/OUTPUT/R_Output/frailty_scores_coxme.csv"
dir.create(dirname(output_path), recursive = TRUE, showWarnings = FALSE)

write_csv(frailty_scores, output_path)
cat("✅ Frailty scores exported to:", output_path, "\n")

# ---- Optional Summary of Variance ----
frailty_var <- as.numeric(VarCorr(coxme_model))
cat("\nEstimated Frailty Variance:", round(frailty_var, 3),
    " | SD =", round(sqrt(frailty_var), 3), "\n")
cat("Hazard multiplier for +1 SD frailty:", round(exp(sqrt(frailty_var)), 2), "x\n")
