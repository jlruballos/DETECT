import pandas as pd
import os
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

program_name = 'fall_gait_sp_plots_v2'

#------ Set up paths ------
base_path = '/mnt/d/DETECT'
base_c_path = '/mnt/d/DETECT_33125/DETECT_Data_Pull_2024-12-16'
output_dir = os.path.join(base_path, 'OUTPUT', program_name)
os.makedirs(output_dir, exist_ok=True)

GAIT_PATH = os.path.join(base_path, 'OUTPUT', 'GAIT', 'COMBINED_NYCE_Area_Data_DETECT_GAIT_summary.csv')
CONTEXT_PATH = os.path.join(base_c_path, '_CONTEXT_FILES', 'Study_Home-Subject_Dates_2024-12-16', 'homeids_subids_NYCE.csv')
FALLS_PATH = os.path.join(base_path, 'OUTPUT', 'survey_processing', 'survey_cleaned.csv')
STEPS_PATH = os.path.join(base_path, 'OUTPUT', 'watch_steps_processing', 'watch_steps_cleaned.csv')

# ------ Style updates ------
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 18,
    'axes.labelsize': 16,
    'xtick.labelsize': 13,
    'ytick.labelsize': 13,
    'legend.fontsize': 13,
    'figure.dpi': 400  # higher quality
})

# ------ Load Data ------
gait_data = pd.read_csv(GAIT_PATH, low_memory=False)
mapping_data = pd.read_csv(CONTEXT_PATH, low_memory=False)
falls = pd.read_csv(FALLS_PATH, low_memory=False)
steps_df = pd.read_csv(STEPS_PATH, low_memory=False)

# Remove homeids with multiple subids
mapping_data = mapping_data.groupby('home_id').filter(lambda x: len(x) == 1)
gait_data = pd.merge(gait_data, mapping_data, left_on='homeid', right_on='home_id', how='inner')
gait_data = gait_data.drop(columns=['home_id'])
gait_data = gait_data.rename(columns={'sub_id': 'subid'})

# Convert fall data
falls['cutoff_dates'] = pd.to_datetime(falls['fall1_date']).dt.date
falls['injury'] = falls['FALL1_INJ']

# ------ Helper: remove outliers ------
def remove_outliers(df, column_name):
    Q1 = df[column_name].quantile(0.25)
    Q3 = df[column_name].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return df[(df[column_name] >= lower) & (df[column_name] <= upper)]

# ------ Main Loop ------
unique_subjects = list(set(gait_data['subid']).intersection(set(steps_df['subid'])))
print(f"Processing {len(unique_subjects)} subjects")

for subid in unique_subjects:
    print(f"Processing subject {subid}")

    # ---- Gait data ----
    gait_sub = gait_data[gait_data['subid'] == subid].copy()
    if gait_sub.empty:
        continue
    gait_sub['date'] = pd.to_datetime(gait_sub['start_time']).dt.date
    gait_avg_df = gait_sub.groupby('date')['gait_speed'].mean().reset_index()
    gait_avg_df = remove_outliers(gait_avg_df, 'gait_speed')
    gait_avg_df = gait_avg_df.sort_values('date')

    # Make day start at 0
    gait_avg_df['Day'] = (pd.to_datetime(gait_avg_df['date']) - pd.to_datetime(gait_avg_df['date'].min())).dt.days
    gait_avg_df['MovingAvg'] = gait_avg_df['gait_speed'].rolling(window=7, min_periods=1).mean()

    # ---- Steps data ----
    steps_sub = steps_df[steps_df['subid'] == subid].copy()
    if steps_sub.empty:
        continue
    steps_sub['date'] = pd.to_datetime(steps_sub['date']).dt.date
    steps_sub = steps_sub[steps_sub['steps'].notna() & (steps_sub['steps'] > 0)]
    steps_sub = remove_outliers(steps_sub, 'steps')
    steps_sub = steps_sub.sort_values('date')

    steps_sub['Day'] = (pd.to_datetime(steps_sub['date']) - pd.to_datetime(steps_sub['date'].min())).dt.days
    steps_sub['MovingAvg'] = steps_sub['steps'].rolling(window=7, min_periods=1).mean()

    # ---- Falls ----
    subject_falls = falls[falls['subid'] == subid].copy()
    if subject_falls.empty:
        continue
    subject_falls['cutoff_dates'] = pd.to_datetime(subject_falls['cutoff_dates'])

    # ==========================================================
    #                 PLOT 1: STEPS
    # ==========================================================
    try:
        fig, ax = plt.subplots(figsize=(14, 6))

        ax.plot(steps_sub['Day'], steps_sub['steps'], '.', color='lightblue', alpha=0.6, label='Daily Steps')
        ax.plot(steps_sub['Day'], steps_sub['MovingAvg'], color='blue', linewidth=2.5, label='7-day Avg Steps')

        # --- Plot fall events ---
        fall_injury_added = False
        fall_noinjury_added = False

        for _, row in subject_falls.iterrows():
            if pd.notna(row['cutoff_dates']):
                fall_date = row['cutoff_dates'].date()
                if fall_date in list(steps_sub['date']):
                    day_idx = steps_sub.loc[steps_sub['date'] == fall_date, 'Day'].values[0]
                    if row['injury'] == 1.0:
                        label = "Fall (Injury)" if not fall_injury_added else None
                        ax.axvline(x=day_idx, color='orange', linestyle='-', linewidth=2, label=label)
                        fall_injury_added = True
                    else:
                        label = "Fall (No Injury)" if not fall_noinjury_added else None
                        ax.axvline(x=day_idx, color='gray', linestyle='-', linewidth=2, label=label)
                        fall_noinjury_added = True

        ax.set_title(f"Subject {subid} - Steps Over Time (Day 0 = first data day)")
        ax.set_xlabel("Day")
        ax.set_ylabel("Steps")
        ax.legend(loc='upper right', frameon=True)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{subid}_steps_plot.png"), dpi=400, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved steps plot for subject {subid}")

    except Exception as e:
        print(f"Error plotting steps for {subid}: {e}")

    # ==========================================================
    #                 PLOT 2: GAIT SPEED
    # ==========================================================
    try:
        fig, ax = plt.subplots(figsize=(14, 6))

        ax.plot(gait_avg_df['Day'], gait_avg_df['gait_speed'], '.', color='lightcoral', alpha=0.6, label='Daily Gait Speed')
        ax.plot(gait_avg_df['Day'], gait_avg_df['MovingAvg'], color='red', linewidth=2.5, label='7-day Avg Gait')

        # --- Plot fall events ---
        fall_injury_added = False
        fall_noinjury_added = False

        for _, row in subject_falls.iterrows():
            if pd.notna(row['cutoff_dates']):
                fall_date = row['cutoff_dates'].date()
                if fall_date in list(gait_avg_df['date']):
                    day_idx = gait_avg_df.loc[gait_avg_df['date'] == fall_date, 'Day'].values[0]
                    if row['injury'] == 1.0:
                        label = "Fall (Injury)" if not fall_injury_added else None
                        ax.axvline(x=day_idx, color='orange', linestyle='-', linewidth=2, label=label)
                        fall_injury_added = True
                    else:
                        label = "Fall (No Injury)" if not fall_noinjury_added else None
                        ax.axvline(x=day_idx, color='gray', linestyle='-', linewidth=2, label=label)
                        fall_noinjury_added = True

        ax.set_title(f"Subject {subid} - Gait Speed Over Time (Day 0 = first data day)")
        ax.set_xlabel("Day")
        ax.set_ylabel("Gait Speed (m/s)")
        ax.legend(loc='upper right', frameon=True)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{subid}_gait_plot.png"), dpi=400, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved gait plot for subject {subid}")

    except Exception as e:
        print(f"Error plotting gait for {subid}: {e}")

print(f"\n✅ Processing complete. All plots saved to {output_dir}")
