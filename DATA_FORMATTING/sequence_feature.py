#!/usr/bin/env python3
"""
Feature Engineering Pipeline for DETECT Sensor Data

This Python script processes longitudinal wearable sensor data for a cohort of older adults as part of a predictive modeling pipeline.
It is designed to support deep learning models (specifically LSTMs) by transforming raw time series data into structured sequences.

The dataset includes daily step counts and in-home gait speed measurements, along with clinical annotations of falls and hospital visits.
Key components of this pipeline include:

- Data loading and validation across multiple sources (gait, steps, clinical events)
- Participant filtering to ensure clean one-to-one mappings (home ID to subject ID)
- Daily-level feature engineering:
  - Normalized values
  - Rolling means (7- and 15-day windows)
  - Deviations from individual baseline (mean-centered deltas)
  - Day-to-day differences (1-day deltas)
- Event labeling (falls and hospitalizations) per day
- Temporal context features:
  - Days since last event
  - Days until next event

Final outputs include a `.csv` file containing full labeled and engineered feature data for each participant.
"""

__author__ = "Jorge Ruballos"
__email__ = "ruballoj@oregonstate.edu"
__date__ = "2025-4-25"
__version__ = "1.0.0"

import pandas as pd
import numpy as np
import os
from datetime import datetime
import sys

# -------- CONFIG --------
USE_WSL = True  # Set to True if running inside WSL (Linux on Windows)
GENERATE_HISTO = True  # Set to True to generate histograms of diagnostic
GENERATE_LINE = True  # Set to True to generate line plots of diagnostic

# Define base paths depending on environment
if USE_WSL:
    base_path = '/mnt/d/DETECT'
else:
    base_path = r'D:\DETECT'

# Add helpers directory to system path
sys.path.append(os.path.join(base_path, 'HELPERS'))
from helpers import (
    preprocess_steps,
    label_exact_day,
    days_since_last_event,
    days_until_next_event,
    impute_missing_data,
    plot_imp_diag_histo,  
    track_missingness,
    plot_imp_diag_timeseries, 
    remove_outliers,
    proc_emfit_data  
)

from vae_imputer import impute_subject_data 


program_name = 'sequence_feature'
imputation_method = 'knn'  # Method for filling missing values
# VAE variant for imputation — choose one of the following:
# 'Zero Imputation': Basic VAE, missing values are replaced with zeros. Mask is not used.
# 'Encoder Mask': Adds the missingness mask as input to the encoder (but not the decoder). Helps the model learn which parts are observed.
# 'Encoder + Decoder Mask': Adds the missingness mask to both encoder and decoder — this is the full corruption-aware variant described in the Collier et al. paper.
variant = 'encoder + decoder mask'

output_path = os.path.join(base_path, 'OUTPUT', program_name)
os.makedirs(output_path, exist_ok=True)  # Create output directory if it does not exist
print(f"Output directory created at: {output_path}")

# File paths for datasets
GAIT_PATH = os.path.join(base_path, 'DETECT_Data', 'NYCE_Data', 'COMBINED_NYCE_Area_Data_DETECT_GAIT_Summary.csv')
STEPS_PATH = os.path.join(base_path, 'DETECT_Data', 'Watch_Data', 'Daily_Steps', 'Watch_Daily_Steps_DETECT_2024-12-16.csv')
MAPPING_PATH = os.path.join(base_path, 'DETECT_Data', '_CONTEXT_FILES', 'Study_Home-Subject_Dates_2024-12-16', 'homeids_subids_NYCE.csv')
FALLS_PATH = os.path.join(base_path, 'DETECT_Data', 'HUF', 'kaye_365_huf_detect.csv')
EMFIT_PATH = os.path.join(base_path, 'DETECT_Data', 'Emfit_Data', 'summary', 'Emfit_Summary_Data_DETECT_2024-12-16.csv')
CLINICAL_PATH = os.path.join(base_path, 'DETECT_Data', 'Clinical', 'Clinical', 'DETECT-AD_Enrolled_Amyloid Status_PET_SUVR_QUEST_CENTILOID_20250116.xlsx')

# -------- LOAD DATA --------
print("Loading data...")
# Load each dataset into DataFrames
gait_df = pd.read_csv(GAIT_PATH)
steps_df = pd.read_csv(STEPS_PATH)
mapping_df = pd.read_csv(MAPPING_PATH)
falls_df = pd.read_csv(FALLS_PATH, low_memory=False)
emfit_df = pd.read_csv(EMFIT_PATH)
clinical_df = pd.read_excel(CLINICAL_PATH)

#-------- EMFIT DATA PREPROCESSING --------
emfit_df = proc_emfit_data(emfit_df)

#-------- REMOVE NEGATIVE VALUES --------
gait_df['gait_speed'] = gait_df['gait_speed'].mask(gait_df['gait_speed'] < 0, np.nan)
steps_df['steps'] = steps_df['steps'].mask(steps_df['steps'] < 0, np.nan)

#--------- FEATURE SELECTION --------
FEATURES = [
    'steps', 
    'gait_speed', 
]

EMFIT_FEATURES = [
       'awakenings', 'bedexitcount', 'end_sleep_time', 
    'inbed_time', 'outbed_time', 'sleepscore', 
    'durationinsleep', 'durationawake', 'waso', 
    'hrvscore', 'start_sleep_time', 'time_to_sleep', 
    'time_in_bed_after_sleep', 'total_time_in_bed', 'tossnturncount', 
    'sleep_period', 'minhr', 'maxhr', 
    'avghr', 'avgrr', 'maxrr', 'minrr'
]

# -------- PREPROCESS --------
# Filter to ensure each home_id maps to only one subid
mapping_data = mapping_df.groupby('home_id').filter(lambda x: len(x) == 1)
mapping_data = mapping_data.rename(columns={'sub_id': 'subid'})

# Merge gait data with mapping to align home IDs with subject IDs
print("Merging gait data with subject mapping...")
gait_df = pd.merge(gait_df, mapping_data, left_on='homeid', right_on='home_id', how='inner')

# Extract date from timestamp and aggregate daily gait speed by subject
gait_df['date'] = pd.to_datetime(gait_df['start_time']).dt.date
gait_df = gait_df.groupby(['subid', 'date'])['gait_speed'].mean().reset_index()

# Preprocess steps data to align with daily format
steps_df = preprocess_steps(steps_df)

# Process fall and hospital event dates
falls_df['cutoff_dates'] = pd.to_datetime(falls_df['fall1_date']).dt.date
falls_df['hospital_visit'] = pd.to_datetime(falls_df['hcru1_date']).dt.date

# -------- FEATURE ENGINEERING --------
all_daily_data = []
missingness_records = []

#mask_features = ['gait_speed', 'daily_steps']  # Features to check for missingness

mask_features = FEATURES + EMFIT_FEATURES  # Features to check for missingness

# Identify subjects who have both gait and step data
#subject_ids = set(gait_df['subid'].unique()).intersection(set(steps_df['subid']))
subject_ids = subject_ids = clinical_df['sub_id'].dropna().unique()
print(f"Number of subjects with gait data: {len(gait_df['subid'].unique())}")

for subid in subject_ids:
    # Extract data for each subject
    gait_sub = gait_df[gait_df['subid'] == subid].copy()
    steps_sub = steps_df[steps_df['subid'] == subid].copy()
    emfit_sub = emfit_df[emfit_df['subid'] == subid].copy()
    subject_falls = falls_df[falls_df['subid'] == subid]
    
    #remove outliers from data
    gait_sub = remove_outliers(gait_sub, 'gait_speed')
    steps_sub = remove_outliers(steps_sub, 'steps')

    if gait_sub.empty and steps_sub.empty and emfit_sub.empty:
        print(f"Skipping subject {subid} due to missing data.")
        continue
 
    # Merge gait and step data by date
    daily_df = pd.merge(gait_sub, steps_sub[['date', 'steps']], on='date', how='outer')
    #make sure dates are of the same type
    daily_df['date'] = pd.to_datetime(daily_df['date']).dt.date
    emfit_sub['date'] = pd.to_datetime(emfit_sub['date']).dt.date
    
    daily_df = pd.merge(daily_df, emfit_sub[['date'] + EMFIT_FEATURES], on='date', how='outer')
    daily_df = daily_df.sort_values('date')
    
    original_daily_df = daily_df.copy()  # Keep a copy of the original DataFrame for reference
    
    #count missing values before imputation
    #missing_before = daily_df[['gait_speed', 'daily_steps']].isna().sum()
    missing_before = daily_df[FEATURES+EMFIT_FEATURES].isna().sum()
    
    #create missingness mask
    missingness_mask = daily_df[mask_features].isna().astype(int) ## 1 for missing, 0 for not missing
    missingness_mask.columns = [col +'_mask' for col in missingness_mask.columns] #rename columns to avoid confusion
    
    if imputation_method == 'vae':
        # Use VAE imputer for missing data
        daily_df = impute_subject_data(daily_df, input_columns=FEATURES+EMFIT_FEATURES, epochs=30, variant=variant)
    else:
		# Fill missing values with specified method
        daily_df = impute_missing_data(daily_df, columns=FEATURES+EMFIT_FEATURES, method=imputation_method, multivariate=False)
    
    #add missingness mask to daily_df
    daily_df = pd.concat([daily_df, missingness_mask], axis=1)
    
    #count missing values after imputation
    missing_after = daily_df[FEATURES+EMFIT_FEATURES].isna().sum()
    
     #Save missingness info to list
    missingness_record = track_missingness(
        original_df=original_daily_df,
        imputed_df=daily_df,
        columns=FEATURES+EMFIT_FEATURES,
        subid=subid,
        imputation_method=imputation_method
    )
    missingness_records.append(missingness_record)
     
    #plot the original daily_df for diagnostics if needed
    if GENERATE_HISTO:
        diagnostic_output_path = os.path.join(output_path, 'diagnostics', imputation_method, 'histograms')
        plot_imp_diag_histo(
			original_df = original_daily_df,
			imputed_df = daily_df,
			columns = FEATURES+EMFIT_FEATURES,
			subid = subid,
			output_dir = diagnostic_output_path,
			method_name = imputation_method
		)
        
    if GENERATE_LINE:
        diagnostic_output_path = os.path.join(output_path, 'diagnostics', imputation_method, 'lineplots')
        plot_imp_diag_timeseries(
			original_df = original_daily_df,
			imputed_df = daily_df,
			columns = FEATURES+EMFIT_FEATURES,
			subid = subid,
			output_dir = diagnostic_output_path,
			method_name = imputation_method
		)
    
    # Create engineered features for gait and steps
    for col in (FEATURES+EMFIT_FEATURES):
        daily_df[col] = pd.to_numeric(daily_df[col], errors='coerce')
        mean = daily_df[col].mean()
        std = daily_df[col].std()
        daily_df[col + '_norm'] = (daily_df[col] - mean) / std  # Normalized feature
        daily_df[col + '_delta'] = daily_df[col] - mean  # Difference from mean
        daily_df[col + '_delta_1d'] = daily_df[col].diff()  # Day-to-day difference
        daily_df[col + '_ma_7'] = daily_df[col].rolling(window=7, min_periods=1).mean()  # 7-day rolling mean

    # Prepare fall and hospital event dates
    fall_dates = sorted(subject_falls['cutoff_dates'].dropna().tolist())
    hospital_dates = sorted(subject_falls['hospital_visit'].dropna().tolist())

    # Create temporal context features
    daily_df['days_since_fall'] = daily_df['date'].apply(lambda d: days_since_last_event(d, fall_dates))
    daily_df['days_until_fall'] = daily_df['date'].apply(lambda d: days_until_next_event(d, fall_dates))
    daily_df['days_since_hospital'] = daily_df['date'].apply(lambda d: days_since_last_event(d, hospital_dates))
    daily_df['days_until_hospital'] = daily_df['date'].apply(lambda d: days_until_next_event(d, hospital_dates))

    # Create event labels
    daily_df['label_fall'] = daily_df['date'].apply(lambda d: label_exact_day(d, fall_dates))
    daily_df['label_hospital'] = daily_df['date'].apply(lambda d: label_exact_day(d, hospital_dates))
    daily_df['label'] = daily_df[['label_fall', 'label_hospital']].max(axis=1)  # Combined label

    daily_df['subid'] = subid  # Add subject ID
    all_daily_data.append(daily_df.copy())
    
    # Print imputation report
    print(f"Subject {subid} - Imputation report ({imputation_method}):")
    for feature in (FEATURES+EMFIT_FEATURES):
    	print(f"  {feature}: {missing_before[feature]} missing → {missing_after[feature]} missing")

# -------- SAVE OUTPUT --------
# Concatenate all subjects' data into one DataFrame
print("Saving labeled daily data...")
final_df = pd.concat(all_daily_data)

# Save final engineered dataset to CSV
csv_filename = f"labeled_daily_data_{imputation_method}.csv"
final_df.to_csv(os.path.join(output_path, csv_filename), index=False)
print("Saved labeled_daily_data.csv with shape:", final_df.shape)

print("Saving missingness report...")
missingness_df = pd.DataFrame(missingness_records)
missingness_report_path = os.path.join(output_path, f"missingness_report_{imputation_method}.csv")
missingness_df.to_csv(missingness_report_path, index=False)
print(f"Saved missingness report to: {missingness_report_path}")
