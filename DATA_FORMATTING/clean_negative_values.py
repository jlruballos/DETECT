import pandas as pd
import os
import shutil

# Set your base path
base_path = '/mnt/d/DETECT/DETECT_Data'

# --- Helper: backup original ---
def backup_file(path):
    backup_path = path.replace('.csv', '_backup.csv')
    shutil.copyfile(path, backup_path)
    print(f"Backup saved to: {backup_path}")

# ----------- CLEAN STEPS -----------
steps_path = os.path.join(base_path, 'Watch_Data', 'Daily_Steps', 'Watch_Daily_Steps_DETECT_2024-12-16.csv')
steps_df = pd.read_csv(steps_path)

backup_file(steps_path)
if 'steps' in steps_df.columns:
    n_neg = (steps_df['steps'] < 0).sum()
    steps_df['steps'] = steps_df['steps'].mask(steps_df['steps'] < 0, pd.NA)
    print(f"Steps: Replaced {n_neg} negative values")

steps_df.to_csv(steps_path, index=False)
print(f"Cleaned steps saved to: {steps_path}")

# ----------- CLEAN GAIT SPEED -----------
gait_path = os.path.join(base_path, 'NYCE_Data', 'COMBINED_NYCE_Area_Data_DETECT_GAIT_Summary.csv')
gait_df = pd.read_csv(gait_path)

backup_file(gait_path)
if 'gait_speed' in gait_df.columns:
    n_neg = (gait_df['gait_speed'] < 0).sum()
    gait_df['gait_speed'] = gait_df['gait_speed'].mask(gait_df['gait_speed'] < 0, pd.NA)
    print(f"Gait: Replaced {n_neg} negative values")

gait_df.to_csv(gait_path, index=False)
print(f"Cleaned gait saved to: {gait_path}")

# ----------- CLEAN EMFIT -----------
emfit_path = os.path.join(base_path, 'Emfit_Data', 'summary', 'Emfit_Summary_Data_DETECT_2024-12-16.csv')
emfit_df = pd.read_csv(emfit_path)

emfit_features = [
    'awakenings', 'bedexitcount', 'end_sleep_time', 'inbed_time', 'outbed_time', 'sleepscore',
    'durationinsleep', 'durationawake', 'waso', 'hrvscore', 'start_sleep_time',
    'time_to_sleep', 'time_in_bed_after_sleep', 'total_time_in_bed', 'tossnturncount',
    'sleep_period', 'minhr', 'maxhr', 'avghr', 'avgrr', 'maxrr', 'minrr'
]

backup_file(emfit_path)

n_total_replaced = 0
for col in emfit_features:
    if col in emfit_df.columns:
        n_neg = (emfit_df[col] < 0).sum()
        emfit_df[col] = emfit_df[col].mask(emfit_df[col] < 0, pd.NA)
        if n_neg > 0:
            print(f"Emfit: Replaced {n_neg} negative values in '{col}'")
        n_total_replaced += n_neg

emfit_df.to_csv(emfit_path, index=False)
print(f"Cleaned Emfit saved to: {emfit_path}")
print(f"Total negative values replaced in Emfit: {n_total_replaced}")
