#!/usr/bin/env python3
"""
Extract and export raw daily-level clinical timeseries features (steps, gait speed, EMFIT)
for use in RStudio imputation workflows like with the `mice` package.
Includes engineered features such as normalized values, deltas, and rolling means.
"""

__author__ = "Jorge Ruballos"
__email__ = "ruballoj@oregonstate.edu"
__date__ = "2025-05-08"
__version__ = "1.1.0"

import pandas as pd
import numpy as np
import os
from datetime import datetime
import sys
from itertools import chain

# -------- CONFIG --------
USE_WSL = True

if USE_WSL:
    base_path = '/mnt/d/DETECT'
else:
    base_path = r'D:\DETECT'

sys.path.append(os.path.join(base_path, 'HELPERS'))
from helpers import preprocess_steps, remove_outliers, proc_emfit_data, label_exact_day, days_since_last_event, days_until_next_event

output_path = os.path.join(base_path, 'OUTPUT', 'raw_export_for_r')
os.makedirs(output_path, exist_ok=True)

# -------- FILE PATHS --------
GAIT_PATH = os.path.join(base_path, 'DETECT_Data', 'NYCE_Data', 'COMBINED_NYCE_Area_Data_DETECT_GAIT_Summary.csv')
STEPS_PATH = os.path.join(base_path, 'DETECT_Data', 'Watch_Data', 'Daily_Steps', 'Watch_Daily_Steps_DETECT_2024-12-16.csv')
MAPPING_PATH = os.path.join(base_path, 'DETECT_Data', '_CONTEXT_FILES', 'Study_Home-Subject_Dates_2024-12-16', 'homeids_subids_NYCE.csv')
EMFIT_PATH = os.path.join(base_path, 'DETECT_Data', 'Emfit_Data', 'summary', 'Emfit_Summary_Data_DETECT_2024-12-16.csv')
CLINICAL_PATH = os.path.join(base_path, 'DETECT_Data', 'Clinical', 'Clinical', 'DETECT-AD_Enrolled_Amyloid Status_PET_SUVR_QUEST_CENTILOID_20250116.xlsx')
FALLS_PATH = os.path.join(base_path, 'DETECT_Data', 'HUF', 'kaye_365_huf_detect.csv')
DEMO_PATH = os.path.join(base_path, 'DETECT_Data', 'Clinical', 'Clinical', 'kaye_365_clin_age_at_visit.csv')

# -------- FEATURES --------
FEATURES = ['steps', 'gait_speed']
EMFIT_FEATURES = [
    'awakenings', 'bedexitcount', 'end_sleep_time', 'inbed_time', 'outbed_time',
    'sleepscore', 'durationinsleep', 'durationawake', 'waso', 'hrvscore',
    'start_sleep_time', 'time_to_sleep', 'time_in_bed_after_sleep',
    'total_time_in_bed', 'tossnturncount', 'sleep_period',
    'minhr', 'maxhr', 'avghr', 'avgrr', 'maxrr', 'minrr'
]

CLINICAL_FEATURES = [
    'amyloid'
]    

DEMO_FEATURES = [
	'birthyr', 'sex', 'hispanic', 'race', 'educ', 'livsitua', 'independ', 'residenc', 'alzdis', 'maristat', 'moca_avg', 'cogstat'
]

# -------- LOAD DATA --------
print("Loading datasets...")
gait_df = pd.read_csv(GAIT_PATH)
steps_df = pd.read_csv(STEPS_PATH)
mapping_df = pd.read_csv(MAPPING_PATH)
emfit_df = pd.read_csv(EMFIT_PATH)
clinical_df = pd.read_excel(CLINICAL_PATH)
falls_df = pd.read_csv(FALLS_PATH)
demo_df = pd.read_csv(DEMO_PATH)

demo_df = demo_df.rename(columns={'record_id': 'subid'})
clinical_df = clinical_df.rename(columns={'sub_id': 'subid'})

emfit_df = proc_emfit_data(emfit_df)

gait_df['gait_speed'] = gait_df['gait_speed'].mask(gait_df['gait_speed'] < 0, np.nan)
steps_df['steps'] = steps_df['steps'].mask(steps_df['steps'] < 0, np.nan)

falls_df['fall1_date'] = pd.to_datetime(falls_df['fall1_date'], errors='coerce')
falls_df['fall2_date'] = pd.to_datetime(falls_df['fall2_date'], errors='coerce')
falls_df['fall3_date'] = pd.to_datetime(falls_df['fall3_date'], errors='coerce')
falls_df['start_date'] = pd.to_datetime(falls_df['StartDate'], errors='coerce')
falls_df['cutoff_dates'] = falls_df.apply(
    lambda row: row['fall1_date'].date() if pd.notnull(row['fall1_date']) 
    else ((row['start_date'] - pd.Timedelta(days=1)).date() if row['FALL'] == 1 else pd.NaT),
    axis=1
)

#first hosptial visit date
falls_df['hospital_visit'] = pd.to_datetime(falls_df['hcru1_date'], errors='coerce').dt.date
falls_df['hospital_visit_2'] = pd.to_datetime(falls_df['hcru2_date'], errors='coerce').dt.date

#combine hospital visit dates into a single column
falls_df['hospital_dates'] = falls_df.apply(
	lambda row: sorted(
		[date for date in [row['hospital_visit'], row['hospital_visit_2']] if pd.notnull(date)]
	), axis=1
)

#import accident date
falls_df['ACDT1_DATE'] = pd.to_datetime(falls_df['acdt1_date'], errors='coerce').dt.date
falls_df['ACDT2_DATE'] = pd.to_datetime(falls_df['acdt2_date'], errors='coerce').dt.date
falls_df['ACDT3_DATE'] = pd.to_datetime(falls_df['acdt3_date'], errors='coerce').dt.date

#combine accident dates into a single column
falls_df['accident_dates'] = falls_df.apply(
	lambda row: sorted(
		[date for date in [row['ACDT1_DATE'], row['ACDT2_DATE'], row['ACDT3_DATE']] if pd.notnull(date)]
	), axis=1
)

#import medication change date
falls_df['MED1_DATE'] = pd.to_datetime(falls_df['med1_date'], errors='coerce').dt.date
falls_df['MED2_DATE'] = pd.to_datetime(falls_df['med2_date'], errors='coerce').dt.date
falls_df['MED3_DATE'] = pd.to_datetime(falls_df['med3_date'], errors='coerce').dt.date
falls_df['MED4_DATE'] = pd.to_datetime(falls_df['med4_date'], errors='coerce').dt.date

#combine accident dates into a single column
falls_df['medication_dates'] = falls_df.apply(
    lambda row: sorted(
		[date for date in [row['MED1_DATE'], row['MED2_DATE'], row['MED3_DATE'], row['MED4_DATE']] if pd.notnull(date)]
	), axis=1
)

#import mood blue data "Have you felt downhearted or blue for three or more days in the past week?" 1: yes 2: no
falls_df['mood_blue_date'] = falls_df.apply(
	lambda row: row['start_date'] - pd.Timedelta(days=1) if row['MOOD_BLUE'] == 1 else pd.NaT, axis=1
)

#import mood lonely "In the past week I felt lonely." 1: yes 2: no
falls_df['mood_lonely_date'] = falls_df.apply(
    lambda row: row['start_date'] - pd.Timedelta(days=1) if row['MOOD_LONV'] == 1 else pd.NaT, axis=1
)

#change amalyoid status to 1 and 0 if Positve chane to 1 and if negative change to 0
clinical_df['amyloid'] = clinical_df['clinical amyloid (+/-) read'].replace({'Positive': 1, 'Negative': 0})

# -------- MERGE AND ALIGN SUBJECTS --------
mapping_data = mapping_df.groupby('home_id').filter(lambda x: len(x) == 1)
mapping_data = mapping_data.rename(columns={'sub_id': 'subid'})
gait_df = pd.merge(gait_df, mapping_data, left_on='homeid', right_on='home_id', how='inner')

gait_df['date'] = pd.to_datetime(gait_df['start_time']).dt.date
gait_df = gait_df.groupby(['subid', 'date'])['gait_speed'].mean().reset_index()
steps_df = preprocess_steps(steps_df)

# -------- EXPORT PER SUBJECT --------
subject_ids = clinical_df['subid'].dropna().unique()
all_subject_data = []


# Filter demo_df to only relevant subjects before MOCA averaging
demo_df = demo_df[demo_df['subid'].isin(subject_ids)]

# Process MOCA scores: average per subject
demo_df['moca_avg'] = demo_df.groupby('subid')['mocatots'].transform('mean')

print(f"Found {len(subject_ids)} subjects with clinical data.")

for subid in subject_ids:
    gait_sub = gait_df[gait_df['subid'] == subid].copy()
    steps_sub = steps_df[steps_df['subid'] == subid].copy()
    emfit_sub = emfit_df[emfit_df['subid'] == subid].copy()
    clinical_sub =  clinical_df[clinical_df['subid'] == subid].copy()
    demo_sub = demo_df[demo_df['subid'] == subid].copy()
    subject_falls = falls_df[falls_df['subid'] == subid]
    fall_dates = sorted(subject_falls['cutoff_dates'].dropna().tolist())
    hospital_dates = sorted(
    list(chain.from_iterable(subject_falls['hospital_dates'].dropna()))
    )
    mood_blue_dates = sorted(
    [d.date() for d in subject_falls['mood_blue_date'].dropna()]
    )
    mood_lonely_dates = sorted(
    [d.date() for d in subject_falls['mood_lonely_date'].dropna()]
    )
    accident_dates = sorted(
    list(chain.from_iterable(subject_falls['accident_dates'].dropna()))
    )
    medication_dates = sorted(
    list(chain.from_iterable(subject_falls['medication_dates'].dropna()))
    )

    gait_sub = remove_outliers(gait_sub, 'gait_speed')
    steps_sub = remove_outliers(steps_sub, 'steps')

    if gait_sub.empty and steps_sub.empty and emfit_sub.empty:
        continue

    daily_df = pd.merge(gait_sub, steps_sub[['date', 'steps']], on='date', how='outer')
    daily_df['date'] = pd.to_datetime(daily_df['date']).dt.date
    emfit_sub['date'] = pd.to_datetime(emfit_sub['date']).dt.date
    daily_df = pd.merge(daily_df, emfit_sub[['date'] + EMFIT_FEATURES], on='date', how='outer')
    daily_df = daily_df.sort_values('date')

	# Add clinical and demographic features
    if not clinical_sub.empty:
        for feat in CLINICAL_FEATURES:
            daily_df[feat] = clinical_sub.iloc[0][feat]
   
    if not demo_sub.empty:
        for feat in DEMO_FEATURES:
            daily_df[feat] = demo_sub.iloc[0][feat]
            
    # Add engineered features efficiently
    feature_blocks = []
    for col in FEATURES + EMFIT_FEATURES:
        daily_df[col] = pd.to_numeric(daily_df[col], errors='coerce')
        mean = daily_df[col].mean()
        std = daily_df[col].std()
        block = pd.DataFrame({
            col + '_norm': (daily_df[col] - mean) / std if std else np.nan,
            col + '_delta': daily_df[col] - mean,
            col + '_delta_1d': daily_df[col].diff(),
            # 7-day backward-looking moving average (includes partial windows for first 6 days)
            col + '_ma_7': daily_df[col].rolling(window=7, min_periods=1).mean()
        })
        feature_blocks.append(block)

    engineered_df = pd.concat(feature_blocks, axis=1)
    daily_df = pd.concat([daily_df.reset_index(drop=True), engineered_df.reset_index(drop=True)], axis=1)

    # Add binary event labels
    daily_df['label_fall'] = daily_df['date'].apply(lambda d: label_exact_day(d, fall_dates))
    daily_df['label_hospital'] = daily_df['date'].apply(lambda d: label_exact_day(d, hospital_dates))
    daily_df['label_mood_blue'] = daily_df['date'].apply(lambda d: label_exact_day(d, mood_blue_dates))
    daily_df['label_mood_lonely'] = daily_df['date'].apply(lambda d: label_exact_day(d, mood_lonely_dates))
    daily_df['label_accident'] = daily_df['date'].apply(lambda d: label_exact_day(d, accident_dates))
    daily_df['label_medication'] = daily_df['date'].apply(lambda d: label_exact_day(d, medication_dates))
    daily_df['label'] = daily_df[['label_fall', 'label_hospital']].max(axis=1)

    # Add temporal context features
    daily_df['days_since_fall'] = daily_df['date'].apply(lambda d: days_since_last_event(d, fall_dates))
    #daily_df['days_until_fall'] = daily_df['date'].apply(lambda d: days_until_next_event(d, fall_dates))
    daily_df['days_since_hospital'] = daily_df['date'].apply(lambda d: days_since_last_event(d, hospital_dates))
    #daily_df['days_until_hospital'] = daily_df['date'].apply(lambda d: days_until_next_event(d, hospital_dates))
    daily_df['days_since_mood_blue'] = daily_df['date'].apply(lambda d: days_since_last_event(d, mood_blue_dates))
    daily_df['days_since_mood_lonely'] = daily_df['date'].apply(lambda d: days_since_last_event(d, mood_lonely_dates))
    daily_df['days_since_accident'] = daily_df['date'].apply(lambda d: days_since_last_event(d, accident_dates))
    daily_df['days_since_medication'] = daily_df['date'].apply(lambda d: days_since_last_event(d, medication_dates))

    daily_df['subid'] = subid
    all_subject_data.append(daily_df)

# -------- SAVE FINAL CSV FOR R --------
print("Saving raw timeseries export for RStudio...")
final_df = pd.concat(all_subject_data, ignore_index=True)
raw_csv_path = os.path.join(output_path, 'raw_daily_data_all_subjects.csv')
final_df.to_csv(raw_csv_path, index=False)
print(f"Exported raw dataset for R to: {raw_csv_path}")