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
from itertools import chain
from scipy.stats import circmean
import matplotlib.pyplot as plt

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
    impute_missing_data,
    plot_imp_diag_histo,  
    track_missingness,
    plot_imp_diag_timeseries, 
    remove_outliers,
    proc_emfit_data  
)

from vae_imputer import impute_subject_data 


program_name = 'sequence_feature'
imputation_method = 'mean'  # Method for filling missing values
# VAE variant for imputation — choose one of the following:
# 'Zero Imputation': Basic VAE, missing values are replaced with zeros. Mask is not used.
# 'Encoder Mask': Adds the missingness mask as input to the encoder (but not the decoder). Helps the model learn which parts are observed.
# 'Encoder + Decoder Mask': Adds the missingness mask to both encoder and decoder — this is the full corruption-aware variant described in the Collier et al. paper.
variant = 'encoder + decoder mask'

output_path = os.path.join(base_path, 'OUTPUT', program_name)
os.makedirs(output_path, exist_ok=True)  # Create output directory if it does not exist

# Create output directory for sleep time plots
sleep_plot_dir = os.path.join(output_path, 'diagnostics', 'sleep_time_plots')
os.makedirs(sleep_plot_dir, exist_ok=True)

print(f"Output directory created at: {output_path}")

# File paths for datasets
GAIT_PATH = os.path.join(base_path, 'DETECT_Data', 'NYCE_Data', 'COMBINED_NYCE_Area_Data_DETECT_GAIT_Summary.csv')
STEPS_PATH = os.path.join(base_path, 'DETECT_Data', 'Watch_Data', 'Daily_Steps', 'Watch_Daily_Steps_DETECT_2024-12-16.csv')
MAPPING_PATH = os.path.join(base_path, 'DETECT_Data', '_CONTEXT_FILES', 'Study_Home-Subject_Dates_2024-12-16', 'homeids_subids_NYCE.csv')
FALLS_PATH = os.path.join(base_path, 'DETECT_Data', 'HUF', 'kaye_365_huf_detect.csv')
EMFIT_PATH = os.path.join(base_path, 'DETECT_Data', 'Emfit_Data', 'summary', 'Emfit_Summary_Data_DETECT_2024-12-16.csv')
CLINICAL_PATH = os.path.join(base_path, 'DETECT_Data', 'Clinical', 'Clinical', 'DETECT-AD_Enrolled_Amyloid Status_PET_SUVR_QUEST_CENTILOID_20250116.xlsx')
DEMO_PATH = os.path.join(base_path, 'DETECT_Data', 'Clinical', 'Clinical', 'kaye_365_clin_age_at_visit.csv')
ACTIVITY_PATH = os.path.join(base_path, 'DETECT_Data', 'Processed_NYCE_Data', 'daily_area_hours_combined_output.csv')

# -------- LOAD DATA --------
print("Loading data...")
# Load each dataset into DataFrames
gait_df = pd.read_csv(GAIT_PATH)
steps_df = pd.read_csv(STEPS_PATH)
mapping_df = pd.read_csv(MAPPING_PATH)
falls_df = pd.read_csv(FALLS_PATH, low_memory=False)
emfit_df = pd.read_csv(EMFIT_PATH)
clinical_df = pd.read_excel(CLINICAL_PATH)
demo_df = pd.read_csv(DEMO_PATH)
activity_df = pd.read_csv(ACTIVITY_PATH)

#--------RENAME COLUMNS --------
demo_df = demo_df.rename(columns={'record_id': 'subid'})
clinical_df = clinical_df.rename(columns={'sub_id': 'subid'})

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
       'awakenings', 'bedexitcount', 
    'inbed_time', 'outbed_time', 'sleepscore', 
    'durationinsleep', 'durationawake', 'waso', 
    'hrvscore', 'time_to_sleep', 
    'time_in_bed_after_sleep', 'total_time_in_bed', 'tossnturncount', 
    'sleep_period', 'minhr', 'maxhr', 
    'avghr', 'avgrr', 'maxrr', 'minrr',
    
    #start and end sleep times have been converted to sin and cosine for easier imputation
    'start_sleep_time_sin', 'start_sleep_time_cos', 'end_sleep_time_sin', 'end_sleep_time_cos'
]

CLINICAL_FEATURES = [
    'amyloid'
] 

DEMO_FEATURES = [
	'birthyr', 'sex', 'hispanic', 'race', 'educ', 'livsitua', 'independ', 'residenc', 'alzdis', 'maristat', 'moca_avg', 'cogstat'
]
#visits are for night time activity 9pm - 6am
ACTIVITY_FEATURES = [
	'Night_Bathroom_Visits', 'Night_Kitchen_Visits'
]

# -------- PREPROCESS --------
#change amalyoid status to 1 and 0 if Positve chane to 1 and if negative change to 0
clinical_df['amyloid'] = clinical_df['clinical amyloid (+/-) read'].replace({'Positive': 1, 'Negative': 0})

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

# Ensure date columns are datetime
falls_df['fall1_date'] = pd.to_datetime(falls_df['fall1_date'], errors='coerce')
falls_df['fall2_date'] = pd.to_datetime(falls_df['fall2_date'], errors='coerce')
falls_df['fall3_date'] = pd.to_datetime(falls_df['fall3_date'], errors='coerce')
falls_df['start_date'] = pd.to_datetime(falls_df['StartDate'], errors='coerce')

# Fill missing fall1_date with (survey_date - 1 day) if fall == 1
falls_df['fall1_date'] = falls_df.apply(
    lambda row: row['fall1_date'].date() if pd.notnull(row['fall1_date']) 
    else ((row['start_date'] - pd.Timedelta(days=1)).date() if row['FALL'] == 1 else pd.NaT),
    axis=1
)
# Process fall and hospital event dates
falls_df['hospital_visit'] = pd.to_datetime(falls_df['hcru1_date']).dt.date
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

# -------- FEATURE ENGINEERING --------
all_daily_data = []
missingness_records = []

#mask_features = ['gait_speed', 'daily_steps']  # Features to check for missingness

mask_features = FEATURES + EMFIT_FEATURES + ACTIVITY_FEATURES  # Features to check for missingness

# Identify subjects who have both gait and step data
#subject_ids = set(gait_df['subid'].unique()).intersection(set(steps_df['subid']))
subject_ids = clinical_df['subid'].dropna().unique()

# Filter demo_df to only relevant subjects before MOCA averaging
demo_df = demo_df[demo_df['subid'].isin(subject_ids)]

# Process MOCA scores: average per subject
demo_df['moca_avg'] = demo_df.groupby('subid')['mocatots'].transform('mean')

print(f"Number of subjects with gait data: {len(gait_df['subid'].unique())}")

# Create output directory for raw data before imputation
raw_output_path = os.path.join(output_path, 'raw_before_imputation')
os.makedirs(raw_output_path, exist_ok=True)
    
for subid in subject_ids:
    # Extract data for each subject
    gait_sub = gait_df[gait_df['subid'] == subid].copy()
    steps_sub = steps_df[steps_df['subid'] == subid].copy()
    emfit_sub = emfit_df[emfit_df['subid'] == subid].copy()
    clinical_sub =  clinical_df[clinical_df['subid'] == subid].copy()
    activity_sub = activity_df[activity_df['subid'] == subid].copy()
    demo_sub = demo_df[demo_df['subid'] == subid].copy()
    subject_falls = falls_df[falls_df['subid'] == subid]
    
    #remove outliers from data
    #gait_sub = remove_outliers(gait_sub, 'gait_speed')
    #steps_sub = remove_outliers(steps_sub, 'steps')

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
    
    # Save raw daily data before imputation (for debugging or audit)
    daily_df.to_csv(os.path.join(raw_output_path, f"{subid}_raw_before_imputation.csv"), index=False)
    print(f"Raw data for subject {subid} saved to {raw_output_path}")
    
    if not activity_sub.empty:
        activity_sub['date'] = pd.to_datetime(activity_sub['Date']).dt.date
        daily_df = pd.merge(daily_df, activity_sub[['date'] + ACTIVITY_FEATURES], on='date', how='outer')
    else:
        print(f"No activity data for subject {subid}, skipping activity features.")
        for feat in ACTIVITY_FEATURES:
            daily_df[feat] = np.nan
    
    original_daily_df = daily_df.copy()
    
    	# Add clinical and demographic features
    if not clinical_sub.empty:
        for feat in CLINICAL_FEATURES:
            daily_df[feat] = clinical_sub.iloc[0][feat]
   
    if not demo_sub.empty:
        for feat in DEMO_FEATURES:
            daily_df[feat] = demo_sub.iloc[0][feat]
    
    #---Compute % missing per festure before imputation
    total_days = len(original_daily_df)
    missing_pct_columns = {}

    for feat in FEATURES+EMFIT_FEATURES+ACTIVITY_FEATURES:
        missing_count = original_daily_df[feat].isna().sum()
        missing_pct = round((missing_count / total_days) * 100, 2) if total_days > 0 else np.nan
        missing_pct_columns[f"{feat}_missing_pct"] = missing_pct
        
    #repeat missing % per features across all rows
    for col_name, value in missing_pct_columns.items():
        daily_df[col_name] = value 
    
    #count missing values before imputation
    #missing_before = daily_df[['gait_speed', 'daily_steps']].isna().sum()
    missing_before = daily_df[FEATURES+EMFIT_FEATURES+ACTIVITY_FEATURES].isna().sum()

    #create missingness mask
    missingness_mask = daily_df[mask_features].isna().astype(int) ## 1 for missing, 0 for not missing
    missingness_mask.columns = [col +'_mask' for col in missingness_mask.columns] #rename columns to avoid confusion
    
    if imputation_method == 'vae':
        # Use VAE imputer for missing data
        daily_df = impute_subject_data(daily_df, input_columns=FEATURES+EMFIT_FEATURES+ACTIVITY_FEATURES, epochs=30, variant=variant)
    else:
		# Fill missing values with specified method
        daily_df = impute_missing_data(daily_df, columns=FEATURES+EMFIT_FEATURES+ACTIVITY_FEATURES, method=imputation_method, multivariate=False)

	# Inverse circular encoding for start and end sleep times
    if imputation_method == 'mean':
        for prefix in ['start_sleep_time', 'end_sleep_time']:
            sin_col = f"{prefix}_sin"
            cos_col = f"{prefix}_cos"
            if sin_col in daily_df.columns and cos_col in daily_df.columns:
                
                # Calculate circular mean for start and end sleep times
				# Drop NaN values for sin/cos columns
                valid = daily_df[[sin_col, cos_col]].dropna()
				# 👉 Use circular mean:
                angles = np.arctan2(
					daily_df[sin_col], daily_df[cos_col]
				)
                circ_mean = np.arctan2(np.nanmean(np.sin(angles)), np.nanmean(np.cos(angles)))

				# Convert back to sin/cos
                sin_mean = np.sin(circ_mean)
                cos_mean = np.cos(circ_mean)

				# Fill missing sin/cos consistently
                missing_mask = daily_df[[sin_col, cos_col]].isna().any(axis=1)
                daily_df.loc[missing_mask, sin_col] = sin_mean
                daily_df.loc[missing_mask, cos_col] = cos_mean
                
                # Quick check
                valid = daily_df[[sin_col, cos_col]].dropna()
                observed_angles = np.arctan2(valid[sin_col], valid[cos_col])
                print(f"{prefix} circmean (hours): {(circmean(observed_angles, high=2*np.pi) * 24 / (2*np.pi)):.2f}")

	# Inverse circular encoding to recover hour values from sin/cos
    for prefix in ['start_sleep_time', 'end_sleep_time']:
        sin_col = f'{prefix}_sin'
        cos_col = f'{prefix}_cos'
        recon_col = prefix  # Save as original name again

		# Only reconstruct if both components are present (and not all missing)
        if sin_col in daily_df.columns and cos_col in daily_df.columns:
            daily_df[recon_col] = (
				np.arctan2(daily_df[sin_col], daily_df[cos_col]) * 24 / (2 * np.pi)
			) % 24
            
        # Plot and save histograms for imputed sleep times
        if GENERATE_HISTO:
            for sleep_col in ['start_sleep_time', 'end_sleep_time']:
                if sleep_col in daily_df.columns:
                    plt.figure()
                    daily_df[sleep_col].hist(bins=24)
                    plt.title(f"{subid} - Imputed {sleep_col.replace('_', ' ').title()}")
                    plt.xlabel("Hour of Day")
                    plt.ylabel("Frequency")
                    plt.grid(True)
                    plt.tight_layout()

					# Save to file
                    plot_path = os.path.join(sleep_plot_dir, f"{subid}_{sleep_col}.png")
                    plt.savefig(plot_path)
                    plt.close()

    #add missingness mask to daily_df
    daily_df = pd.concat([daily_df, missingness_mask], axis=1)
    
    #count missing values after imputation
    missing_after = daily_df[FEATURES+EMFIT_FEATURES+ACTIVITY_FEATURES].isna().sum()

     #Save missingness info to list
    missingness_record = track_missingness(
        original_df=original_daily_df,
        imputed_df=daily_df,
        columns=FEATURES+EMFIT_FEATURES + ACTIVITY_FEATURES,
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
			columns = FEATURES+EMFIT_FEATURES + ACTIVITY_FEATURES,
			subid = subid,
			output_dir = diagnostic_output_path,
			method_name = imputation_method
		)
        
    if GENERATE_LINE:
        diagnostic_output_path = os.path.join(output_path, 'diagnostics', imputation_method, 'lineplots')
        plot_imp_diag_timeseries(
			original_df = original_daily_df,
			imputed_df = daily_df,
			columns = FEATURES+EMFIT_FEATURES + ACTIVITY_FEATURES,
			subid = subid,
			output_dir = diagnostic_output_path,
			method_name = imputation_method
		)
    
    # Create engineered features for gait and steps
    engineered_cols = {}
    for col in (FEATURES + EMFIT_FEATURES + ACTIVITY_FEATURES):
        mean = daily_df[col].mean()
        std = daily_df[col].std()
        engineered_cols[col + '_norm'] = (daily_df[col] - mean) / std
        engineered_cols[col + '_delta'] = daily_df[col] - mean
        engineered_cols[col + '_delta_1d'] = daily_df[col].diff()
        engineered_cols[col + '_ma_7'] = daily_df[col].rolling(window=7, min_periods=1).mean()
    
    daily_df = pd.concat([daily_df, pd.DataFrame(engineered_cols)], axis=1)
    
    daily_df = daily_df.copy()

    # Prepare fall and hospital event dates
    raw_fall_dates = list(chain.from_iterable([
		subject_falls['fall1_date'].dropna().tolist(),
		subject_falls['fall2_date'].dropna().tolist(),
		subject_falls['fall3_date'].dropna().tolist()
	]))
    
    fall_dates = sorted(set([pd.to_datetime(d).date() for d in raw_fall_dates]))

    
    hospital_dates = sorted(
    list(chain.from_iterable(
        [x for x in subject_falls['hospital_dates'] if isinstance(x, list) and x]
    ))
    )
    mood_blue_dates = sorted(
    [d.date() for d in subject_falls['mood_blue_date'].dropna()]
    )
    mood_lonely_dates = sorted(
    [d.date() for d in subject_falls['mood_lonely_date'].dropna()]
    )
    
    accident_dates = sorted(
    list(chain.from_iterable(
        [x for x in subject_falls['accident_dates'] if isinstance(x, list) and x]
    ))
)
    medication_dates = sorted(
    list(chain.from_iterable(
        [x for x in subject_falls['medication_dates'] if isinstance(x, list) and x]
    ))
    )

    # Create temporal context features
    daily_df['days_since_fall'] = daily_df['date'].apply(lambda d: days_since_last_event(d, fall_dates))
    daily_df['days_since_hospital'] = daily_df['date'].apply(lambda d: days_since_last_event(d, hospital_dates))
    daily_df['days_since_mood_blue'] = daily_df['date'].apply(lambda d: days_since_last_event(d, mood_blue_dates))
    daily_df['days_since_mood_lonely'] = daily_df['date'].apply(lambda d: days_since_last_event(d, mood_lonely_dates))
    daily_df['days_since_accident'] = daily_df['date'].apply(lambda d: days_since_last_event(d, accident_dates))
    daily_df['days_since_medication'] = daily_df['date'].apply(lambda d: days_since_last_event(d, medication_dates))

    # Create event labels
    daily_df['label_fall'] = daily_df['date'].apply(lambda d: label_exact_day(d, fall_dates))
    daily_df['label_hospital'] = daily_df['date'].apply(lambda d: label_exact_day(d, hospital_dates))
    daily_df['label_mood_blue'] = daily_df['date'].apply(lambda d: label_exact_day(d, mood_blue_dates))
    daily_df['label_mood_lonely'] = daily_df['date'].apply(lambda d: label_exact_day(d, mood_lonely_dates))
    daily_df['label_accident'] = daily_df['date'].apply(lambda d: label_exact_day(d, accident_dates))
    daily_df['label_medication'] = daily_df['date'].apply(lambda d: label_exact_day(d, medication_dates))
    daily_df['label'] = daily_df[['label_fall', 'label_hospital', 'label_mood_blue', 'label_mood_lonely', 'label_accident', 'label_medication']].max(axis=1)  # Combined label

    daily_df['subid'] = subid  # Add subject ID
    all_daily_data.append(daily_df.copy())
    
    # Print imputation report
    print(f"Subject {subid} - Imputation report ({imputation_method}):")
    for feature in (FEATURES+EMFIT_FEATURES+ ACTIVITY_FEATURES):
    	print(f"  {feature}: {missing_before[feature]} missing → {missing_after[feature]} missing")

#----------REORDERING COLUMNS----------
# Concatenate all subjects' data into one DataFrame
print("Saving labeled daily data...")
final_df = pd.concat(all_daily_data)

# -------- REORDER COLUMNS --------
temporal_cols = (
    FEATURES + EMFIT_FEATURES + ACTIVITY_FEATURES +
    [f"{f}_norm" for f in FEATURES + EMFIT_FEATURES + ACTIVITY_FEATURES] +
    [f"{f}_delta" for f in FEATURES + EMFIT_FEATURES + ACTIVITY_FEATURES] +
    [f"{f}_delta_1d" for f in FEATURES + EMFIT_FEATURES + ACTIVITY_FEATURES] +
    [f"{f}_ma_7" for f in FEATURES + EMFIT_FEATURES + ACTIVITY_FEATURES]
)

# Remove sine/cosine columns explicitly if present
temporal_cols = [col for col in temporal_cols if col not in [
    'start_sleep_time_sin', 'start_sleep_time_cos',
    'end_sleep_time_sin', 'end_sleep_time_cos'
]]

#make sure to add the converted start and end sleep times
temporal_cols += ['start_sleep_time', 'end_sleep_time']

final_df = final_df[
    ['subid', 'date'] + temporal_cols + CLINICAL_FEATURES + DEMO_FEATURES +
    ['days_since_fall',  'days_since_hospital', 'days_since_mood_blue', 'days_since_mood_lonely', 'days_since_accident', 'days_since_medication'] +
     ['label_fall', 'label_hospital', 'label', 'label_mood_blue', 'label_mood_lonely', 'label_accident', 'label_medication']
]

# -------- SAVE OUTPUT --------

# Save final engineered dataset to CSV
csv_filename = f"labeled_daily_data_{imputation_method}.csv"
final_df.to_csv(os.path.join(output_path, csv_filename), index=False)
print("Saved labeled_daily_data.csv with shape:", final_df.shape)

print("Saving missingness report...")
missingness_df = pd.DataFrame(missingness_records)
missingness_report_path = os.path.join(output_path, f"missingness_report_{imputation_method}.csv")
missingness_df.to_csv(missingness_report_path, index=False)
print(f"Saved missingness report to: {missingness_report_path}")
