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

# -------- CONFIG --------
USE_WSL = True  # Set to True if running inside WSL (Linux on Windows)
GENERATE_HISTO = False  # Set to True to generate histograms of diagnostic
GENERATE_LINE = False # Set to True to generate line plots of diagnostic

# Define base paths depending on environment
if USE_WSL:
    base_path = '/mnt/d/DETECT'
else:
    base_path = r'D:\DETECT'

# Add helpers directory to system path
sys.path.append(os.path.join(base_path, 'HELPERS'))
from helpers import (
    label_exact_day,
    days_since_last_event,
    proc_emfit_data  
)

program_name = 'combine_data'

output_path = os.path.join(base_path, 'OUTPUT', program_name)
os.makedirs(output_path, exist_ok=True)  # Create output directory if it does not exist

print(f"Output directory created at: {output_path}")

# Log file for the pipeline
# -------- LOGGING --------
# add timestamp to log file name
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
logfile = os.path.join(output_path, f"pipeline_log_{timestamp}.txt")
def log_step(message):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_msg = f"[{ts}] {message}"
    print(full_msg)
    with open(logfile, "a") as f:
        f.write(full_msg + "\n")

# File paths for datasets
GAIT_PATH = os.path.join(base_path, 'OUTPUT', 'gait_speed_processing', 'gait_speed_cleaned.csv')
STEPS_PATH = os.path.join(base_path, 'OUTPUT', 'watch_steps_processing','watch_steps_cleaned.csv')
MAPPING_PATH = os.path.join(base_path, 'DETECT_Data', '_CONTEXT_FILES', 'Study_Home-Subject_Dates_2024-12-16', 'homeids_subids_NYCE.csv')
FALLS_PATH = os.path.join(base_path, 'OUTPUT', 'survey_processing', 'survey_cleaned.csv')
EMFIT_PATH = os.path.join(base_path, 'OUTPUT', 'emfit_processing', 'emfit_sleep_1.csv')
CLINICAL_PATH = os.path.join(base_path, 'OUTPUT', 'clinical_participant_processing', 'clinical_cleaned.csv')
DEMO_PATH = os.path.join(base_path, 'OUTPUT', 'yearly_visit_processing', 'yearly_recoded.csv')
ACTIVITY_PATH = os.path.join(base_path, 'OUTPUT', 'activity_processing', 'activity_cleaned.csv')

# -------- LOAD DATA --------
print("Loading data...")
# Load each dataset into DataFrames
gait_df = pd.read_csv(GAIT_PATH)
steps_df = pd.read_csv(STEPS_PATH)
mapping_df = pd.read_csv(MAPPING_PATH)
falls_df = pd.read_csv(FALLS_PATH, low_memory=False)
emfit_df = pd.read_csv(EMFIT_PATH)
clinical_df = pd.read_csv(CLINICAL_PATH)
demo_df = pd.read_csv(DEMO_PATH)
activity_df = pd.read_csv(ACTIVITY_PATH)

# Filter to ensure each home_id maps to only one subid
mapping_data = mapping_df.groupby('home_id').filter(lambda x: len(x) == 1)
mapping_data = mapping_data.rename(columns={'sub_id': 'subid'})

#--------RENAME COLUMNS --------
demo_df = demo_df.rename(columns={'record_id': 'subid'})
clinical_df = clinical_df.rename(columns={'sub_id': 'subid'})

#-------- ROW COUNTS --------
log_step(f"[INITIAL] Loaded gait_df: N = {len(gait_df)}, Unique subids = {gait_df['subid'].nunique()}")
log_step(f"[INITIAL] Loaded steps_df: N = {len(steps_df)}, Unique subids = {steps_df['subid'].nunique()}")
log_step(f"[INITIAL] Loaded falls_df: N = {len(falls_df)}, Unique subids = {falls_df['subid'].nunique()}")
log_step(f"[INITIAL] Loaded emfit_df: N = {len(emfit_df)}, Unique subids = {emfit_df['subid'].nunique()}")
log_step(f"[INITIAL] Loaded clinical_df: N = {len(clinical_df)}, Unique subids = {clinical_df['subid'].nunique()}")
log_step(f"[INITIAL] Loaded demo_df: N = {len(demo_df)}, Unique subids = {demo_df['subid'].nunique()}")
log_step(f"[INITIAL] Loaded activity_df: N = {len(activity_df)}, Unique subids = {activity_df['subid'].nunique()}")

#-------- EMFIT DATA PREPROCESSING --------
emfit_df = proc_emfit_data(emfit_df)

#--------- FEATURE SELECTION --------
FEATURES = [
    'steps', 
    'gait_speed', 
]

EMFIT_FEATURES = [
       'awakenings', 'bedexitcount', 
    'inbed_time', 'outbed_time', 'sleepscore', 
    'durationinsleep', 'durationawake', 'waso', 
    'hrvscore', 
    'time_in_bed_after_sleep', 'total_time_in_bed', 'tossnturncount', 
    'sleep_period', 'minhr', 'maxhr', 
    'avghr', 'avgrr', 'maxrr', 'minrr',
    'start_sleep_time', 'end_sleep_time',
]

CLINICAL_FEATURES = [
    'amyloid'
] 

DEMO_FEATURES = [
	'birthyr', 'sex', 'hispanic', 'race', 'educ', 'livsitua', 'independ', 'residenc', 'alzdis', 'maristat', 'moca_avg', 'cogstat',
    'primlang', 'mocatots', 'age_at_visit', 'age_bucket', 'educ_group', 'moca_category', 'race_group'
]
#visits are for night time activity 9pm - 6am
ACTIVITY_FEATURES = [
	'Night_Bathroom_Visits', 'Night_Kitchen_Visits'
]

# -------- FEATURE ENGINEERING --------
all_daily_data = []
missingness_records = []
all_daily_data_raw = []  # Store raw daily data before imputation

#mask_features = ['gait_speed', 'daily_steps']  # Features to check for missingness

mask_features = FEATURES + EMFIT_FEATURES + ACTIVITY_FEATURES  # Features to check for missingness

# Get unique subids from all major datasets
subid_sets = [
    set(gait_df['subid'].dropna().unique()),
    set(steps_df['subid'].dropna().unique()),
    set(falls_df['subid'].dropna().unique()),
    set(emfit_df['subid'].dropna().unique()),
    set(activity_df['subid'].dropna().unique()),
    set(demo_df['subid'].dropna().unique()),
    set(clinical_df['subid'].dropna().unique())  # optional, still include clinical info if present
]

# Take the union of all subids
subject_ids = sorted(set.union(*subid_sets))

log_step(f"Total subids after union across all datasets: {len(subject_ids)}")

#log what subids were dropped
dropped_subids_gait = set(gait_df['subid'].unique()).difference(set(subject_ids))
log_step(f"Dropped subids from gait_df: {dropped_subids_gait}")
dropped_subids_steps = set(steps_df['subid'].unique()).difference(set(subject_ids))
log_step(f"Dropped subids from steps_df: {dropped_subids_steps}")
dropped_subids_falls = set(falls_df['subid'].unique()).difference(set(subject_ids))
log_step(f"Dropped subids from falls_df: {dropped_subids_falls}")
dropped_subids_emfit = set(emfit_df['subid'].unique()).difference(set(subject_ids))
log_step(f"Dropped subids from emfit_df: {dropped_subids_emfit}")
dropped_subids_clinical = set(clinical_df['subid'].unique()).difference(set(subject_ids))
log_step(f"Dropped subids from clinical_df: {dropped_subids_clinical}")
dropped_subids_demo = set(demo_df['subid'].unique()).difference(set(subject_ids))
log_step(f"Dropped subids from demo_df: {dropped_subids_demo}")
dropped_subids_activity = set(activity_df['subid'].unique()).difference(set(subject_ids))
log_step(f"Dropped subids from activity_df: {dropped_subids_activity}")

# Create output directory for raw data before imputation
raw_output_path = os.path.join(output_path, 'raw_before_imputation')
os.makedirs(raw_output_path, exist_ok=True)

# -------- SUMMARY TRACKERS --------
summary_rows = []
summary_missing = []
    
for subid in subject_ids:

    # Extract data for each subject
    gait_sub = gait_df[gait_df['subid'] == subid].copy()
    steps_sub = steps_df[steps_df['subid'] == subid].copy()
    emfit_sub = emfit_df[emfit_df['subid'] == subid].copy()
    clinical_sub =  clinical_df[clinical_df['subid'] == subid].copy()
    activity_sub = activity_df[activity_df['subid'] == subid].copy()
    #demo_sub = demo_df[demo_df['subid'] == subid].copy()
    subject_falls = falls_df[falls_df['subid'] == subid]

    if gait_sub.empty and steps_sub.empty and emfit_sub.empty:
        print(f"Skipping subject {subid} due to missing data.")
        continue
 
    # Merge gait and step data by date
    daily_df = pd.merge(gait_sub, steps_sub[['date', 'steps']], on='date', how='outer')
    #check for duplicates after merging gait and steps
    n_dupes = daily_df.duplicated(subset=['subid', 'date']).sum()
    if n_dupes > 0:
        log_step(f"WARNING: {n_dupes} duplicate rows found for {subid} in gait and steps merge!")
    
    #orphan row check
    feature_cols_temp = ['gait_speed', 'steps']
    all_nan_rows = daily_df[feature_cols_temp].isna().all(axis=1).sum()
    if all_nan_rows > 0:
        log_step(f"WARNING: {all_nan_rows} rows with all feature columns NaN for {subid} in gait and steps merge!")

    #log number of rows in daily_df after merging gait and steps
    log_step(f"Subject {subid} - Daily Data after merging gait and steps: N = {len(daily_df)}")
    
    #make sure dates are of the same type
    daily_df['date'] = pd.to_datetime(daily_df['date']).dt.date
    emfit_sub['date'] = pd.to_datetime(emfit_sub['date']).dt.date
    
    daily_df = pd.merge(daily_df, emfit_sub[['date'] + EMFIT_FEATURES], on='date', how='outer')
    
	#check for duplicates after merging emfit data
    n_dupes = daily_df.duplicated(subset=['subid', 'date']).sum()
    if n_dupes > 0:
        log_step(f"WARNING: {n_dupes} duplicate rows found for {subid} in emfit merge!")
    
    #orphan row check
    emfit_features_temp = EMFIT_FEATURES
    all_nan_rows = daily_df[emfit_features_temp].isna().all(axis=1).sum()
    if all_nan_rows > 0:
        log_step(f"WARNING: {all_nan_rows} rows with all emfit feature columns NaN for {subid} in emfit merge!")
    
    daily_df = daily_df.sort_values('date')
    
    #log number of rows in daily_df after merging emfit data
    log_step(f"Subject {subid} - Daily Data after merging emfit data: N = {len(daily_df)}")
    
    if not activity_sub.empty:
        activity_sub['date'] = pd.to_datetime(activity_sub['Date']).dt.date
        daily_df = pd.merge(daily_df, activity_sub[['date'] + ACTIVITY_FEATURES], on='date', how='outer')
        
        #check for duplicates after merging activity data
        n_dupes = daily_df.duplicated(subset=['subid', 'date']).sum()
        if n_dupes > 0:
            log_step(f"WARNING: {n_dupes} duplicate rows found for {subid} in activity merge!")
        
        #orphan row check
        activity_features_temp = ACTIVITY_FEATURES
        all_nan_rows = daily_df[activity_features_temp].isna().all(axis=1).sum()
        if all_nan_rows > 0:
            log_step(f"WARNING: {all_nan_rows} rows with all activity feature columns NaN for {subid} in activity merge!")

        #log number of rows in daily_df after merging activity data
        log_step(f"Subject {subid} - Daily Data after merging activity data: N = {len(daily_df)}")
    else:
        print(f"No activity data for subject {subid}, skipping activity features.")
        for feat in ACTIVITY_FEATURES:
            daily_df[feat] = np.nan
    
    	# Add clinical and demographic features
    if not clinical_sub.empty:
        for feat in CLINICAL_FEATURES:
            daily_df[feat] = clinical_sub.iloc[0][feat]
            #log number of rows in daily_df after adding clinical features
            log_step(f"Subject {subid} - Daily Data after adding clinical features: N = {len(daily_df)}")

    #Assign time-vaying demo features by year, used for demo features
    daily_df['year'] = pd.to_datetime(daily_df['date']).dt.year
    daily_df['year_date'] = pd.to_datetime(daily_df['year'].astype(str) + '-01-01')  # Create a date for the first day of the year
    daily_df = daily_df.sort_values('year_date') # Sort by year_date to ensure chronological order
    
    subject_demo = demo_df[demo_df['subid'] == subid].copy()
    #subject_demo['visit_date'] = pd.to_datetime(subject_demo['visityr_cr'].astype(str) + '-01-01')  # Create a date for the first day of the visit year
    subject_demo = subject_demo.sort_values('visit_date')  # Sort by visit date to ensure chronological order
    
    #keep only the most recent record per visit_date
    subject_demo = subject_demo.groupby(['subid', 'visit_date'], as_index=False).last()

    subject_demo['visit_date'] = pd.to_datetime(subject_demo['visit_date'], errors='coerce')
    
    if subject_demo['visit_date'].isna().all():
        print(f"[WARN] All visit dates missing or invalid for subject {subid}")
            
    #merge demo features with backward asof
    daily_df = pd.merge_asof(
        daily_df,
        subject_demo[['visit_date'] + DEMO_FEATURES],
        left_on='year_date', right_on='visit_date',
		direction='backward',  # Use the most recent demo features before the date
    )
    #check for duplicates after merging demo features
    n_dupes = daily_df.duplicated(subset=['subid', 'date']).sum()
    if n_dupes > 0:
        log_step(f"WARNING: {n_dupes} duplicate rows found for {subid} in demo features merge!")
        
    #check for orphan rows after merging demo features
    demo_features_temp = DEMO_FEATURES
    all_nan_rows = daily_df[demo_features_temp].isna().all(axis=1).sum()
    if all_nan_rows > 0:
        log_step(f"WARNING: {all_nan_rows} rows with all demo feature columns NaN for {subid} in demo features merge!")
    
    #log number of rows in daily_df after merging demo features
    log_step(f"Subject {subid} - Daily Data after merging demo features: N = {len(daily_df)}")
    
    #Fill any remaining NaNs (days before first checkup) with the earliest available demo features
    for feat in DEMO_FEATURES:
        if feat in subject_demo and not subject_demo.empty:
            first_val = subject_demo[feat].iloc[0]
            daily_df[feat] = daily_df[feat].fillna(first_val)
    
    #log number of rows in daily_df after filling NaNs with earliest demo features
    log_step(f"Subject {subid} - Daily Data after filling NaNs with earliest demo features: N = {len(daily_df)}")
    
    # --- DUPLICATE CHECK ---
    n_dupes = daily_df.duplicated(subset=['subid', 'date']).sum()
    if n_dupes > 0:
        log_step(f"WARNING: {n_dupes} duplicate rows found for {subid}!")
    
    #check for orphan rows after merging demo features
    demo_features_temp = DEMO_FEATURES
    all_nan_rows = daily_df[demo_features_temp].isna().all(axis=1).sum()
    if all_nan_rows > 0:
        log_step(f"WARNING: {all_nan_rows} rows with all demo feature columns NaN for {subid} in demo features merge!")

    #Drop helper columns
    daily_df = daily_df.drop(columns=['year', 'year_date', 'visit_date'], axis=1, errors='ignore')
    
    #make sure dates are in order
    daily_df = daily_df.sort_values('date').reset_index(drop=True)
    
    #make sure the subid clumn is populated with the current subid
    daily_df['subid'] = subid
    
    #concatinate raw_daily_df to all_daily_data_raw
    all_daily_data_raw.append(daily_df)
    
    #---Compute % missing per festure before
    total_days = len(daily_df)
    missing_pct_columns = {}

    for feat in FEATURES+EMFIT_FEATURES+ACTIVITY_FEATURES:
        missing_count = daily_df[feat].isna().sum()
        missing_pct = round((missing_count / total_days) * 100, 2) if total_days > 0 else np.nan
        missing_pct_columns[f"{feat}_missing_pct"] = missing_pct
        
    #repeat missing % per features across all rows
    for col_name, value in missing_pct_columns.items():
        daily_df[col_name] = value 
    
    #count missing values
    missing_before = daily_df[FEATURES+EMFIT_FEATURES+ACTIVITY_FEATURES].isna().sum()

    #create missingness mask
    missingness_mask = daily_df[mask_features].isna().astype(int) ## 1 for missing, 0 for not missing
    missingness_mask.columns = [col +'_mask' for col in missingness_mask.columns] #rename columns to avoid confusion
    
    #log number of rows in daily_df after creating missingness mask
    log_step(f"Subject {subid} - Daily Data: N = {len(daily_df)}")
    
    #add missingness mask to daily_df
    daily_df = pd.concat([daily_df, missingness_mask], axis=1)
    
    #log number of rows in daily_df after adding missingness mask
    log_step(f"Subject {subid} - Daily Data after adding missingness mask: N = {len(daily_df)}")

    for col in ['mood_blue_date', 'mood_lonely_date', 'fall_date', 'hospital_date', 'accident_date', 'medication_date']:
        if col in subject_falls.columns:
            subject_falls.loc[:, col] = pd.to_datetime(subject_falls[col], errors='coerce')

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
    
    #log number of rows in daily_df after adding temporal context features
    log_step(f"Subject {subid} - Daily Data after adding temporal context features: N = {len(daily_df)}")

    # Create event labels
    daily_df['label_fall'] = daily_df['date'].apply(lambda d: label_exact_day(d, fall_dates))
    daily_df['label_hospital'] = daily_df['date'].apply(lambda d: label_exact_day(d, hospital_dates))
    daily_df['label_mood_blue'] = daily_df['date'].apply(lambda d: label_exact_day(d, mood_blue_dates))
    daily_df['label_mood_lonely'] = daily_df['date'].apply(lambda d: label_exact_day(d, mood_lonely_dates))
    daily_df['label_accident'] = daily_df['date'].apply(lambda d: label_exact_day(d, accident_dates))
    daily_df['label_medication'] = daily_df['date'].apply(lambda d: label_exact_day(d, medication_dates))
    daily_df['label'] = daily_df[['label_fall', 'label_hospital', 'label_mood_blue', 'label_mood_lonely', 'label_accident', 'label_medication']].max(axis=1)  # Combined label
    
    #log number of rows in daily_df after adding event labels
    log_step(f"Subject {subid} - Daily Data after adding event labels: N = {len(daily_df)}")

    daily_df['subid'] = subid  # Add subject ID
    #log number of rows in daily_df after adding subject ID
    log_step(f"Subject {subid} - Daily Data after adding subject ID: N = {len(daily_df)}")
    
    all_daily_data.append(daily_df.copy())
    
    #log the number of rows in all_daily_data after appending subject data
    log_step(f"Appended data for subject {subid}. Total rows in all_daily_data: {len(pd.concat(all_daily_data))}")
    
    # Print imputation report
    for feature in (FEATURES+EMFIT_FEATURES+ ACTIVITY_FEATURES):
        print(f"  {feature}: {missing_before[feature]} missing")
    
    summary_rows.append({
        'subid': subid,
        'n_rows_after_merges': len(daily_df)
    })
    missing_counts = daily_df.isnull().sum()
    missing_pct = (missing_counts / len(daily_df)) * 100
    missing_pct['subid'] = subid
    summary_missing.append(missing_pct)


#----------REORDERING COLUMNS----------
# Concatenate all subjects' data into one DataFrame
print("Saving labeled daily data...")
final_df = pd.concat(all_daily_data)

# Final duplicate check: Is every subid-date combo present only once?
n_dupes = final_df.duplicated(subset=['subid', 'date']).sum()
if n_dupes > 0:
    log_step(f"ERROR: {n_dupes} duplicated subid-date pairs found in final_df after concatenation!")
    # Print out first few duplicate rows for inspection
    print("First 10 duplicated (subid, date) entries:")
    print(final_df[final_df.duplicated(['subid', 'date'], keep=False)].head(10))
    dup_df = final_df[final_df.duplicated(['subid', 'date'], keep=False)]
    dup_df.to_csv(os.path.join(output_path, 'debug_duplicated_subid_date.csv'), index=False)
else:
    log_step("SUCCESS: No duplicated subid-date pairs in final_df.")

#check for duplicates in final_df
n_dupes = final_df.duplicated(subset=['subid', 'date']).sum()
if n_dupes > 0:
	log_step(f"WARNING: {n_dupes} duplicate rows found in final_df!")
 
# --- After concatenating all_daily_data into final_df ---
final_n = final_df['subid'].nunique()
log_step(f"Final output N subjects: {final_n} (should match subject_ids N: {len(subject_ids)})")
if final_n < len(subject_ids):
    log_step(f"WARNING: {len(subject_ids) - final_n} subjects missing from final_df!")
    #print out the missing subids
    missing_subids = set(subject_ids) - set(final_df['subid'].unique())
    print(f"Missing subids: {missing_subids}")
    log_step(f"WARNING: {len(missing_subids)} subjects missing from final_df!")
log_step(f"Shape of final_df: {final_df.shape}")


final_df = final_df[
    ['subid', 'date'] +
    FEATURES +
    CLINICAL_FEATURES +
    EMFIT_FEATURES +
    ACTIVITY_FEATURES +
    DEMO_FEATURES +
    ['days_since_fall', 'days_since_hospital', 'days_since_mood_blue', 'days_since_mood_lonely', 'days_since_accident', 'days_since_medication'] +
    ['label_fall', 'label_hospital', 'label', 'label_mood_blue', 'label_mood_lonely', 'label_accident', 'label_medication']
]

#check for duplicates in final_df before saving to CSV
n_dupes = final_df.duplicated(subset=['subid', 'date']).sum()
if n_dupes > 0:
    log_step(f"WARNING: {n_dupes} duplicate rows found in final_df!")

#check for orphan rows in final_df before saving to CSV
orphan_features = FEATURES + EMFIT_FEATURES + ACTIVITY_FEATURES + DEMO_FEATURES 
existing_orphan_features = [col for col in orphan_features if col in final_df.columns]
all_nan_rows = final_df[existing_orphan_features].isna().all(axis=1).sum()
if all_nan_rows > 0:
	log_step(f"WARNING: {all_nan_rows} rows with all feature columns NaN in final_df before saving to CSV!")

# -------- SAVE OUTPUT --------

# Save final engineered dataset to CSV
csv_filename = f"labeled_daily_data_unimputed.csv"
final_df.to_csv(os.path.join(output_path, csv_filename), index=False)
print("Saved labeled_daily_data_unimputed.csv with shape:", final_df.shape)

# After processing all subjects:
summary_rows_df = pd.DataFrame(summary_rows)
summary_missing_df = pd.DataFrame(summary_missing)
summary_rows_df.to_csv(os.path.join(output_path, 'summary_row_counts_per_subject.csv'), index=False)
summary_missing_df.to_csv(os.path.join(output_path, 'summary_missingness_pct_per_subject.csv'), index=False)
log_step("Saved subject-level row counts and missingness summaries.")

# Final summary:
final_summary = pd.DataFrame({
    'n_subids': [final_df['subid'].nunique()],
    'total_rows': [len(final_df)],
    'n_features': [len(final_df.columns)],
    'n_duplicates': [final_df.duplicated(subset=['subid', 'date']).sum()],
    'n_orphan_rows': [final_df[existing_orphan_features].isna().all(axis=1).sum()]
})
final_summary.to_csv(os.path.join(output_path, 'summary_pipeline_overview.csv'), index=False)
log_step("Saved final pipeline summary.")

# Save final output as before
csv_filename = f"labeled_daily_data_unimputed.csv"
final_df.to_csv(os.path.join(output_path, csv_filename), index=False)
log_step(f"Saved labeled daily data with shape {final_df.shape}")

