#!/usr/bin/env python3
"""
Count and visualize missing values per feature per subid using raw DETECT data (pre-imputation).

Author: Jorge Ruballos
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import sys

# -------- CONFIG --------
USE_WSL = True
base_path = '/mnt/d/DETECT' if USE_WSL else r'D:\DETECT'
program_name = 'count_missing_values'

# File paths
GAIT_PATH = os.path.join(base_path, 'DETECT_Data', 'NYCE_Data', 'COMBINED_NYCE_Area_Data_DETECT_GAIT_Summary.csv')
STEPS_PATH = os.path.join(base_path, 'DETECT_Data', 'Watch_Data', 'Daily_Steps', 'Watch_Daily_Steps_DETECT_2024-12-16.csv')
MAPPING_PATH = os.path.join(base_path, 'DETECT_Data', '_CONTEXT_FILES', 'Study_Home-Subject_Dates_2024-12-16', 'homeids_subids_NYCE.csv')
EMFIT_PATH = os.path.join(base_path, 'DETECT_Data', 'Emfit_Data', 'summary', 'Emfit_Summary_Data_DETECT_2024-12-16.csv')
CLINICAL_PATH = os.path.join(base_path, 'DETECT_Data', 'Clinical', 'Clinical', 'DETECT-AD_Enrolled_Amyloid Status_PET_SUVR_QUEST_CENTILOID_20250116.xlsx')

# Output
output_dir = os.path.join(base_path, 'OUTPUT', program_name, 'missingness_summary_raw')
os.makedirs(output_dir, exist_ok=True)

# Add helper functions
sys.path.append(os.path.join(base_path, 'HELPERS'))
from helpers import preprocess_steps, proc_emfit_data, remove_outliers

# -------- FEATURES --------
FEATURES = ['steps', 'gait_speed']
EMFIT_FEATURES = [
    'awakenings', 'bedexitcount', 'end_sleep_time',
    'inbed_time', 'outbed_time', 'sleepscore',
    'durationinsleep', 'durationawake', 'waso',
    'hrvscore', 'start_sleep_time', 'time_to_sleep',
    'time_in_bed_after_sleep', 'total_time_in_bed', 'tossnturncount',
    'sleep_period', 'minhr', 'maxhr',
    'avghr', 'avgrr', 'maxrr', 'minrr'
]
ALL_FEATURES = FEATURES + EMFIT_FEATURES

# -------- LOAD DATA --------
print("Loading raw datasets...")
gait_df = pd.read_csv(GAIT_PATH)
steps_df = pd.read_csv(STEPS_PATH)
mapping_df = pd.read_csv(MAPPING_PATH)
emfit_df = pd.read_csv(EMFIT_PATH)
clinical_df = pd.read_excel(CLINICAL_PATH)

# -------- PREPROCESSING --------
print("Preprocessing...")

# Handle negatives
gait_df['gait_speed'] = gait_df['gait_speed'].mask(gait_df['gait_speed'] < 0, np.nan)
steps_df['steps'] = steps_df['steps'].mask(steps_df['steps'] < 0, np.nan)

# Emfit
emfit_df = proc_emfit_data(emfit_df)

# Ensure date format
steps_df['date'] = pd.to_datetime(steps_df['date']).dt.date
emfit_df['date'] = pd.to_datetime(emfit_df['date']).dt.date
gait_df['date'] = pd.to_datetime(gait_df['start_time']).dt.date

# Map homeid → subid
mapping_df = mapping_df.groupby('home_id').filter(lambda x: len(x) == 1)
mapping_df = mapping_df.rename(columns={'sub_id': 'subid'})
gait_df = pd.merge(gait_df, mapping_df, left_on='homeid', right_on='home_id', how='inner')
gait_df = gait_df.groupby(['subid', 'date'])['gait_speed'].mean().reset_index()

steps_df = preprocess_steps(steps_df)
subject_ids = clinical_df['sub_id'].dropna().unique()

# -------- ANALYZE MISSINGNESS --------
print("Analyzing missing values...")
missing_records = []

for subid in subject_ids:
    gait_sub = gait_df[gait_df['subid'] == subid].copy()
    steps_sub = steps_df[steps_df['subid'] == subid].copy()
    emfit_sub = emfit_df[emfit_df['subid'] == subid].copy()

    gait_sub = remove_outliers(gait_sub, 'gait_speed')
    steps_sub = remove_outliers(steps_sub, 'steps')

    if gait_sub.empty and steps_sub.empty and emfit_sub.empty:
        continue

    daily_df = pd.merge(gait_sub[['date', 'gait_speed']], steps_sub[['date', 'steps']], on='date', how='outer')
    daily_df = pd.merge(daily_df, emfit_sub[['date'] + EMFIT_FEATURES], on='date', how='outer')
    daily_df = daily_df.sort_values('date')

    total_days = len(daily_df)
    if total_days == 0:
        continue

    missing_counts = daily_df[ALL_FEATURES].isna().sum()
    missing_percents = (missing_counts / total_days * 100).round(2)

    total_possible = total_days * len(ALL_FEATURES)
    total_missing = int(missing_counts.sum())
    total_pct = round((total_missing / total_possible) * 100, 2)

    row = {
        'subid': subid,
        'total_days': total_days,
        'total_possible': total_possible,
        'total_missing': total_missing,
        'total_pct_missing': total_pct
    }
    for feat in ALL_FEATURES:
        row[feat + '_missing'] = int(missing_counts.get(feat, 0))
        row[feat + '_pct'] = float(missing_percents.get(feat, 0))
    missing_records.append(row)
    
    summary_df = pd.DataFrame(missing_records)

	# Confirm these columns exist
    assert 'total_days' in summary_df.columns
    assert 'total_possible' in summary_df.columns
    assert 'total_missing' in summary_df.columns
    assert 'total_pct_missing' in summary_df.columns
    
    summary_df.to_csv(os.path.join(output_dir, 'missing_summary_per_subid.csv'), index=False)
 
    # --- Plot ---
    features_present = [feat for feat in ALL_FEATURES if feat + '_pct' in row]
    pct_values = [row[feat + '_pct'] for feat in features_present]
    plt.figure(figsize=(12, 6))
    plt.bar(features_present, pct_values, color='tomato')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('% Missing')
    plt.title(f'% Missing per Feature - subid {subid} ({total_days} days)')
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'missing_pct_subid_{subid}.png'))
    plt.close()
