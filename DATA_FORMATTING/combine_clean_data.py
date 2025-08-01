#!/usr/bin/env python3
"""
Feature Engineering Pipeline for DETECT Sensor Data with Step Tracking

This Python script processes longitudinal wearable sensor data for a cohort of older adults as part of a predictive modeling pipeline.
It is designed to support deep learning models (specifically LSTMs) by transforming raw time series data into structured sequences.

Enhanced with detailed step-by-step tracking of row counts, missing values, and duplicates at each merge operation.
"""

__author__ = "Jorge Ruballos"
__email__ = "ruballoj@oregonstate.edu"
__date__ = "2025-4-25"
__version__ = "1.1.0"

import pandas as pd
import numpy as np
import os
from datetime import datetime, date
import sys
from itertools import chain
import ast

def parse_datetime_string_list(series, column_name):
    """Parse string representations of datetime.date lists from CSV"""
    all_dates = []
    
    for value in series.dropna():
        try:
            if pd.isna(value) or value == '' or value == '[]':
                continue
            
            # Handle string representation of Python lists with datetime.date objects
            if isinstance(value, str) and value.startswith('['):
                try:
                    # Use ast.literal_eval to safely evaluate the string
                    # But first we need to handle datetime.date() calls
                    
                    # Replace datetime.date(...) with a parseable format
                    import re
                    
                    # Find all datetime.date(year, month, day) patterns
                    date_pattern = r'datetime\.date\((\d+),\s*(\d+),\s*(\d+)\)'
                    
                    def date_replacer(match):
                        year, month, day = match.groups()
                        # Convert to a date object
                        return f"date({year}, {month}, {day})"
                    
                    # Replace datetime.date calls with date calls
                    modified_value = re.sub(date_pattern, date_replacer, value)
                    
                    # Now evaluate, but we need to handle the date() calls
                    # Let's extract the date parameters instead
                    date_matches = re.findall(date_pattern, value)
                    
                    for year_str, month_str, day_str in date_matches:
                        year, month, day = int(year_str), int(month_str), int(day_str)
                        all_dates.append(date(year, month, day))
                        
                except Exception as e:
                    print(f"Error parsing {column_name} string '{value}': {e}")
                    
        except Exception as e:
            print(f"Error processing {column_name} value '{value}': {e}")
    
    return sorted(set(all_dates))

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

program_name = 'combine_clean_data'

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

# -------- STEP TRACKING FUNCTIONS --------
def track_step(df, subid, step_name, step_num, tracking_records):
    """Track metrics for each processing step"""
    n_rows = len(df)
    n_dupes = df.duplicated(subset=['subid', 'date']).sum() if 'date' in df.columns else 0
    
    # Count missing values for key columns
    key_cols = ['gait_speed', 'steps'] + [col for col in df.columns if col in EMFIT_FEATURES + ACTIVITY_FEATURES + DEMO_FEATURES + CLINICAL_FEATURES]
    missing_counts = {}
    for col in key_cols:
        if col in df.columns:
            missing_counts[f'{col}_missing'] = df[col].isna().sum()
    
    # Store tracking info
    record = {
        'subid': subid,
        'step_num': step_num,
        'step_name': step_name,
        'n_rows': n_rows,
        'n_duplicates': n_dupes,
        **missing_counts
    }
    
    tracking_records.append(record)
    
    # Log summary
    missing_summary = ', '.join([f"{k.replace('_missing', '')}: {v}" for k, v in missing_counts.items() if v > 0])
    log_step(f"Subject {subid} - {step_name}: N={n_rows}, Dupes={n_dupes}, Missing=[{missing_summary}]")
    
    return record

# File paths for datasets
GAIT_PATH = os.path.join(base_path, 'OUTPUT', 'gait_speed_processing', 'gait_speed_cleaned.csv')
STEPS_PATH = os.path.join(base_path, 'OUTPUT', 'watch_steps_processing','watch_steps_cleaned.csv')
MAPPING_PATH = os.path.join(base_path, 'DETECT_Data', '_CONTEXT_FILES', 'Study_Home-Subject_Dates_2024-12-16', 'homeids_subids_NYCE.csv')
FALLS_PATH = os.path.join(base_path, 'OUTPUT', 'survey_processing', 'survey_cleaned.csv')
EMFIT_PATH = os.path.join(base_path, 'OUTPUT', 'emfit_processing', 'emfit_sleep_1.csv')
CLINICAL_PATH = os.path.join(base_path, 'OUTPUT', 'clinical_participant_processing', 'clinical_cleaned.csv')
DEMO_PATH = os.path.join(base_path, 'OUTPUT', 'yearly_visit_processing', 'yearly_recoded.csv')
ACTIVITY_PATH = os.path.join(base_path, 'OUTPUT', 'activity_processing', 'activity_cleaned.csv')

# Validate files exist upfront
required_files = [GAIT_PATH, STEPS_PATH, MAPPING_PATH, FALLS_PATH, EMFIT_PATH, CLINICAL_PATH, DEMO_PATH, ACTIVITY_PATH]
for file_path in required_files:
    if not os.path.exists(file_path):
        log_step(f"ERROR: Required file not found: {file_path}")
        sys.exit(1)

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

ACTIVITY_FEATURES = [
	'Night_Bathroom_Visits', 'Night_Kitchen_Visits'
]

# -------- FEATURE ENGINEERING --------
all_daily_data = []
all_daily_data_raw = []  # Store raw daily data before imputation
tracking_records = []  # Track metrics at each step

mask_features = FEATURES + EMFIT_FEATURES + ACTIVITY_FEATURES  # Features to check for missingness

# Get unique subids from all major datasets
subid_sets = [
    set(gait_df['subid'].dropna().unique()),
    set(steps_df['subid'].dropna().unique()),
    set(falls_df['subid'].dropna().unique()),
    set(emfit_df['subid'].dropna().unique()),
    set(activity_df['subid'].dropna().unique()),
    set(demo_df['subid'].dropna().unique()),
    set(clinical_df['subid'].dropna().unique())
]

# Take the union of all subids
subject_ids = sorted(set.union(*subid_sets))

log_step(f"Total subids after union across all datasets: {len(subject_ids)}")

# -------- LOG MISSING SUBIDS --------
# Track which subids are missing from each dataset
dataset_info = {
    'gait': set(gait_df['subid'].dropna().unique()),
    'steps': set(steps_df['subid'].dropna().unique()),
    'falls': set(falls_df['subid'].dropna().unique()),
    'emfit': set(emfit_df['subid'].dropna().unique()),
    'activity': set(activity_df['subid'].dropna().unique()),
    'demo': set(demo_df['subid'].dropna().unique()),
    'clinical': set(clinical_df['subid'].dropna().unique())
}

missing_subids_report = []
for dataset_name, dataset_subids in dataset_info.items():
    missing_from_dataset = set(subject_ids) - dataset_subids
    present_in_dataset = dataset_subids.intersection(set(subject_ids))
    
    log_step(f"Dataset '{dataset_name}': {len(present_in_dataset)} present, {len(missing_from_dataset)} missing")
    
    # Add to report
    for subid in subject_ids:
        missing_subids_report.append({
            'subid': subid,
            'dataset': dataset_name,
            'present': subid in dataset_subids
        })

# Save missing subids report
missing_subids_df = pd.DataFrame(missing_subids_report)
missing_subids_pivot = missing_subids_df.pivot(index='subid', columns='dataset', values='present').fillna(False)
missing_subids_pivot.to_csv(os.path.join(output_path, 'subids_presence_by_dataset.csv'))
log_step("Saved subids presence report by dataset.")

# -------- SUMMARY TRACKERS --------
summary_rows = []
summary_missing = []
skipped_subjects = []  # Track subjects that were skipped
    
for subid in subject_ids:
    log_step(f"\n=== Processing Subject {subid} ===")
    
    # Extract data for each subject
    gait_sub = gait_df[gait_df['subid'] == subid].copy()
    steps_sub = steps_df[steps_df['subid'] == subid].copy()
    emfit_sub = emfit_df[emfit_df['subid'] == subid].copy()
    clinical_sub = clinical_df[clinical_df['subid'] == subid].copy()
    activity_sub = activity_df[activity_df['subid'] == subid].copy()
    subject_falls = falls_df[falls_df['subid'] == subid]

    if gait_sub.empty and steps_sub.empty and emfit_sub.empty:
        log_step(f"Skipping subject {subid} due to missing data.")
        skipped_subjects.append({
            'subid': subid,
            'reason': 'missing_core_data',
            'gait_present': not gait_sub.empty,
            'steps_present': not steps_sub.empty,
            'emfit_present': not emfit_sub.empty,
            'activity_present': not activity_sub.empty,
            'clinical_present': not clinical_sub.empty,
            'demo_present': subid in set(demo_df['subid'].unique()),
            'falls_present': not subject_falls.empty
        })
        continue

    # STEP 1: Merge gait and step data
    daily_df = pd.merge(gait_sub, steps_sub[['date', 'steps']], on='date', how='outer')
    daily_df['subid'] = subid  # Ensure subid is populated
    track_step(daily_df, subid, "after_gait_steps_merge", 1, tracking_records)
    
    # STEP 2: Merge emfit data
    daily_df['date'] = pd.to_datetime(daily_df['date']).dt.date
    emfit_sub['date'] = pd.to_datetime(emfit_sub['date']).dt.date
    daily_df = pd.merge(daily_df, emfit_sub[['date'] + EMFIT_FEATURES], on='date', how='outer')
    daily_df = daily_df.sort_values('date')
    daily_df['subid'] = subid  # Ensure subid is populated after merge
    track_step(daily_df, subid, "after_emfit_merge", 2, tracking_records)
    
    # STEP 3: Merge activity data
    if not activity_sub.empty:
        activity_sub['date'] = pd.to_datetime(activity_sub['Date']).dt.date
        daily_df = pd.merge(daily_df, activity_sub[['date'] + ACTIVITY_FEATURES], on='date', how='outer')
        daily_df['subid'] = subid  # Ensure subid is populated after merge
        track_step(daily_df, subid, "after_activity_merge", 3, tracking_records)
    else:
        log_step(f"No activity data for subject {subid}, filling with NaN")
        for feat in ACTIVITY_FEATURES:
            daily_df[feat] = np.nan
        track_step(daily_df, subid, "after_activity_fill_nan", 3, tracking_records)
    
    # STEP 4: Add clinical features
    if not clinical_sub.empty:
        for feat in CLINICAL_FEATURES:
            if feat in clinical_sub.columns:
                daily_df[feat] = clinical_sub.iloc[0][feat]
        track_step(daily_df, subid, "after_clinical_features", 4, tracking_records)
    else:
        log_step(f"No clinical data for subject {subid}, filling with NaN")
        for feat in CLINICAL_FEATURES:
            daily_df[feat] = np.nan
        track_step(daily_df, subid, "after_clinical_fill_nan", 4, tracking_records)

    # STEP 5: Add demographic features (time-varying by year)
    daily_df['year'] = pd.to_datetime(daily_df['date']).dt.year
    daily_df['year_date'] = pd.to_datetime(daily_df['year'].astype(str) + '-01-01')
    daily_df = daily_df.sort_values('year_date')
    
    subject_demo = demo_df[demo_df['subid'] == subid].copy()
    if not subject_demo.empty:
        subject_demo = subject_demo.sort_values('visit_date')
        subject_demo = subject_demo.groupby(['subid', 'visit_date'], as_index=False).last()
        subject_demo['visit_date'] = pd.to_datetime(subject_demo['visit_date'], errors='coerce')
        
        # Merge demo features with backward asof
        daily_df = pd.merge_asof(
            daily_df,
            subject_demo[['visit_date'] + DEMO_FEATURES],
            left_on='year_date', right_on='visit_date',
            direction='backward',
        )
        
        # Fill any remaining NaNs with earliest available demo features
        for feat in DEMO_FEATURES:
            if feat in subject_demo.columns and not subject_demo.empty:
                first_val = subject_demo[feat].iloc[0]
                daily_df[feat] = daily_df[feat].fillna(first_val)
    else:
        log_step(f"No demographic data for subject {subid}, filling with NaN")
        for feat in DEMO_FEATURES:
            daily_df[feat] = np.nan
            
    daily_df['subid'] = subid  # Ensure subid is populated after merge
    track_step(daily_df, subid, "after_demo_features", 5, tracking_records)

    # Drop helper columns
    daily_df = daily_df.drop(columns=['year', 'year_date', 'visit_date'], errors='ignore')
    daily_df = daily_df.sort_values('date').reset_index(drop=True)
    daily_df['subid'] = subid
    
    # STEP 6: Store raw data before imputation
    all_daily_data_raw.append(daily_df.copy())
    
    # STEP 7: Compute missingness percentages and create masks
    total_days = len(daily_df)
    missing_pct_columns = {}

    for feat in FEATURES + EMFIT_FEATURES + ACTIVITY_FEATURES:
        if feat in daily_df.columns:
            missing_count = daily_df[feat].isna().sum()
            missing_pct = round((missing_count / total_days) * 100, 2) if total_days > 0 else np.nan
            missing_pct_columns[f"{feat}_missing_pct"] = missing_pct
        
    # Add missing percentages to dataframe
    for col_name, value in missing_pct_columns.items():
        daily_df[col_name] = value 

    # Create missingness mask
    available_mask_features = [col for col in mask_features if col in daily_df.columns]
    missingness_mask = daily_df[available_mask_features].isna().astype(int)
    missingness_mask.columns = [col +'_mask' for col in missingness_mask.columns]
    daily_df = pd.concat([daily_df, missingness_mask], axis=1)
    
    track_step(daily_df, subid, "after_missingness_processing", 6, tracking_records)

    # STEP 8: Process fall and event data
    for col in ['mood_blue_date', 'mood_lonely_date']:
        if col in subject_falls.columns:
            subject_falls.loc[:, col] = pd.to_datetime(subject_falls[col], errors='coerce')

    # Prepare event dates
    raw_fall_dates = list(chain.from_iterable([
        subject_falls['fall1_date'].dropna().tolist(),
        subject_falls['fall2_date'].dropna().tolist(),
        subject_falls['fall3_date'].dropna().tolist()
    ]))
    
    fall_dates = sorted(set([pd.to_datetime(d).date() for d in raw_fall_dates]))
    mood_blue_dates = sorted([d.date() for d in subject_falls['mood_blue_date'].dropna()])
    mood_lonely_dates = sorted([d.date() for d in subject_falls['mood_lonely_date'].dropna()])

    hospital_dates = parse_datetime_string_list(
    subject_falls['hospital_dates'] if 'hospital_dates' in subject_falls.columns else pd.Series([]), 
    'hospital_dates'
    )

    accident_dates = parse_datetime_string_list(
        subject_falls['accident_dates'] if 'accident_dates' in subject_falls.columns else pd.Series([]), 
        'accident_dates'
    )

    medication_dates = parse_datetime_string_list(
        subject_falls['medication_dates'] if 'medication_dates' in subject_falls.columns else pd.Series([]), 
        'medication_dates'
    )

    # logging to verify
    log_step(f"Subject {subid} - Parsed dates: hospital={len(hospital_dates)}, accident={len(accident_dates)}, medication={len(medication_dates)}")
    if hospital_dates:
        log_step(f"  Sample hospital dates: {hospital_dates[:3]}")
    if accident_dates:
        log_step(f"  Sample accident dates: {accident_dates[:3]}")
    if medication_dates:
        log_step(f"  Sample medication dates: {medication_dates[:3]}")

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
    daily_df['label'] = daily_df[['label_fall', 'label_hospital', 'label_mood_blue', 'label_mood_lonely', 'label_accident', 'label_medication']].max(axis=1)

    track_step(daily_df, subid, "after_event_processing", 7, tracking_records)
    
    all_daily_data.append(daily_df.copy())
    
    # Summary for this subject
    summary_rows.append({
        'subid': subid,
        'n_rows_final': len(daily_df)
    })

#----------FINAL PROCESSING----------
print("Saving labeled daily data...")
final_df = pd.concat(all_daily_data, ignore_index=True)

# Final duplicate check
n_dupes = final_df.duplicated(subset=['subid', 'date']).sum()
if n_dupes > 0:
    log_step(f"ERROR: {n_dupes} duplicated subid-date pairs found in final_df!")
    dup_df = final_df[final_df.duplicated(['subid', 'date'], keep=False)]
    dup_df.to_csv(os.path.join(output_path, 'debug_duplicated_subid_date.csv'), index=False)
else:
    log_step("SUCCESS: No duplicated subid-date pairs in final_df.")

# Safely reorder columns - only include columns that actually exist
available_cols = ['subid', 'date']
col_groups = [
    FEATURES,
    CLINICAL_FEATURES, 
    EMFIT_FEATURES,
    ACTIVITY_FEATURES,
    DEMO_FEATURES,
    ['days_since_fall', 'days_since_hospital', 'days_since_mood_blue', 'days_since_mood_lonely', 'days_since_accident', 'days_since_medication'],
    ['label_fall', 'label_hospital', 'label', 'label_mood_blue', 'label_mood_lonely', 'label_accident', 'label_medication']
]

for col_group in col_groups:
    available_cols.extend([col for col in col_group if col in final_df.columns])

# Log any missing expected columns
all_expected_cols = set(['subid', 'date'] + sum(col_groups, []))
missing_cols = all_expected_cols - set(final_df.columns)
if missing_cols:
    log_step(f"WARNING: Expected columns not found in final_df: {missing_cols}")

final_df = final_df[available_cols]

# -------- SAVE OUTPUT --------
csv_filename = "labeled_daily_data_unimputed.csv"
final_df.to_csv(os.path.join(output_path, csv_filename), index=False)
log_step(f"Saved labeled daily data with shape {final_df.shape}")

# Save tracking records
tracking_df = pd.DataFrame(tracking_records)
tracking_df.to_csv(os.path.join(output_path, 'step_by_step_tracking.csv'), index=False)
log_step("Saved step-by-step tracking data.")

# Save skipped subjects report
if skipped_subjects:
    skipped_df = pd.DataFrame(skipped_subjects)
    skipped_df.to_csv(os.path.join(output_path, 'skipped_subjects_report.csv'), index=False)
    log_step(f"Saved report of {len(skipped_subjects)} skipped subjects.")
else:
    log_step("No subjects were skipped.")

# -------- FINAL MISSING SUBIDS ANALYSIS --------
# Check which subjects from the original union made it to the final dataset
final_subids = set(final_df['subid'].unique())
original_subids = set(subject_ids)
processed_subids = set([record['subid'] for record in summary_rows])
truly_missing_subids = original_subids - final_subids

missing_subids_final = []
for subid in original_subids:
    status = 'processed'
    if subid in [s['subid'] for s in skipped_subjects]:
        status = 'skipped_no_core_data'
    elif subid not in final_subids:
        status = 'missing_from_final'
    
    missing_subids_final.append({
        'subid': subid,
        'status': status,
        'in_final_dataset': subid in final_subids
    })

missing_final_df = pd.DataFrame(missing_subids_final)
missing_final_df.to_csv(os.path.join(output_path, 'final_subids_status_report.csv'), index=False)

log_step(f"Final subids analysis: {len(final_subids)} in final dataset, {len(truly_missing_subids)} missing from final")
if truly_missing_subids:
    log_step(f"Subids missing from final dataset: {sorted(truly_missing_subids)}")

# Save summary reports
summary_rows_df = pd.DataFrame(summary_rows)
summary_rows_df.to_csv(os.path.join(output_path, 'summary_row_counts_per_subject.csv'), index=False)

# Final summary
final_summary = pd.DataFrame({
    'n_subids': [final_df['subid'].nunique()],
    'total_rows': [len(final_df)],
    'n_features': [len(final_df.columns)],
    'n_duplicates': [final_df.duplicated(subset=['subid', 'date']).sum()]
})
final_summary.to_csv(os.path.join(output_path, 'summary_pipeline_overview.csv'), index=False)
log_step("Saved final pipeline summary.")

print(f"\nPipeline completed successfully!")
print(f"Final dataset: {final_df.shape[0]} rows, {final_df.shape[1]} columns, {final_df['subid'].nunique()} subjects")