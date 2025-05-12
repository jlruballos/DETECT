#!/usr/bin/env python3
"""
Extract and export raw daily-level clinical timeseries features (steps, gait speed, EMFIT)
for use in RStudio imputation workflows like with the `mice` package.
"""

__author__ = "Jorge Ruballos"
__email__ = "ruballoj@oregonstate.edu"
__date__ = "2025-05-08"
__version__ = "1.0.0"

import pandas as pd
import numpy as np
import os
from datetime import datetime
import sys

# -------- CONFIG --------
USE_WSL = True

if USE_WSL:
    base_path = '/mnt/d/DETECT'
else:
    base_path = r'D:\DETECT'

sys.path.append(os.path.join(base_path, 'HELPERS'))
from helpers import preprocess_steps, remove_outliers, proc_emfit_data

output_path = os.path.join(base_path, 'OUTPUT', 'raw_export_for_r')
os.makedirs(output_path, exist_ok=True)

# -------- FILE PATHS --------
GAIT_PATH = os.path.join(base_path, 'DETECT_Data', 'NYCE_Data', 'COMBINED_NYCE_Area_Data_DETECT_GAIT_Summary.csv')
STEPS_PATH = os.path.join(base_path, 'DETECT_Data', 'Watch_Data', 'Daily_Steps', 'Watch_Daily_Steps_DETECT_2024-12-16.csv')
MAPPING_PATH = os.path.join(base_path, 'DETECT_Data', '_CONTEXT_FILES', 'Study_Home-Subject_Dates_2024-12-16', 'homeids_subids_NYCE.csv')
EMFIT_PATH = os.path.join(base_path, 'DETECT_Data', 'Emfit_Data', 'summary', 'Emfit_Summary_Data_DETECT_2024-12-16.csv')
CLINICAL_PATH = os.path.join(base_path, 'DETECT_Data', 'Clinical', 'Clinical', 'DETECT-AD_Enrolled_Amyloid Status_PET_SUVR_QUEST_CENTILOID_20250116.xlsx')

# -------- FEATURES --------
FEATURES = ['steps', 'gait_speed']
EMFIT_FEATURES = [
    'awakenings', 'bedexitcount', 'end_sleep_time', 'inbed_time', 'outbed_time',
    'sleepscore', 'durationinsleep', 'durationawake', 'waso', 'hrvscore',
    'start_sleep_time', 'time_to_sleep', 'time_in_bed_after_sleep',
    'total_time_in_bed', 'tossnturncount', 'sleep_period',
    'minhr', 'maxhr', 'avghr', 'avgrr', 'maxrr', 'minrr'
]

# -------- LOAD DATA --------
print("Loading datasets...")
gait_df = pd.read_csv(GAIT_PATH)
steps_df = pd.read_csv(STEPS_PATH)
mapping_df = pd.read_csv(MAPPING_PATH)
emfit_df = pd.read_csv(EMFIT_PATH)
clinical_df = pd.read_excel(CLINICAL_PATH)

emfit_df = proc_emfit_data(emfit_df)

gait_df['gait_speed'] = gait_df['gait_speed'].mask(gait_df['gait_speed'] < 0, np.nan)
steps_df['steps'] = steps_df['steps'].mask(steps_df['steps'] < 0, np.nan)

# -------- MERGE AND ALIGN SUBJECTS --------
mapping_data = mapping_df.groupby('home_id').filter(lambda x: len(x) == 1)
mapping_data = mapping_data.rename(columns={'sub_id': 'subid'})
gait_df = pd.merge(gait_df, mapping_data, left_on='homeid', right_on='home_id', how='inner')

gait_df['date'] = pd.to_datetime(gait_df['start_time']).dt.date
gait_df = gait_df.groupby(['subid', 'date'])['gait_speed'].mean().reset_index()
steps_df = preprocess_steps(steps_df)

# -------- EXPORT PER SUBJECT --------
subject_ids = clinical_df['sub_id'].dropna().unique()
all_subject_data = []

print(f"Found {len(subject_ids)} subjects with clinical data.")

for subid in subject_ids:
    gait_sub = gait_df[gait_df['subid'] == subid].copy()
    steps_sub = steps_df[steps_df['subid'] == subid].copy()
    emfit_sub = emfit_df[emfit_df['subid'] == subid].copy()

    gait_sub = remove_outliers(gait_sub, 'gait_speed')
    steps_sub = remove_outliers(steps_sub, 'steps')

    if gait_sub.empty and steps_sub.empty and emfit_sub.empty:
        continue

    daily_df = pd.merge(gait_sub, steps_sub[['date', 'steps']], on='date', how='outer')
    daily_df['date'] = pd.to_datetime(daily_df['date']).dt.date
    emfit_sub['date'] = pd.to_datetime(emfit_sub['date']).dt.date
    daily_df = pd.merge(daily_df, emfit_sub[['date'] + EMFIT_FEATURES], on='date', how='outer')
    daily_df = daily_df.sort_values('date')
    daily_df['subid'] = subid

    all_subject_data.append(daily_df)

# -------- SAVE FINAL CSV FOR R --------
print("Saving raw timeseries export for RStudio...")
final_df = pd.concat(all_subject_data, ignore_index=True)
raw_csv_path = os.path.join(output_path, 'raw_daily_data_all_subjects.csv')
final_df.to_csv(raw_csv_path, index=False)
print(f"Exported raw dataset for R to: {raw_csv_path}")