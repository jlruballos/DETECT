#!/usr/bin/env python3
"""
Imputation Pipeline for DETECT Sensor Data

This script takes the unimputed dataset from the combine_clean_data pipeline
and applies various imputation methods with feature engineering.

The script supports:
- Multiple imputation methods (mean, median, forward fill, VAE)
- Circular encoding handling for sleep times
- Feature engineering (normalization, deltas, rolling means)
- Comprehensive tracking and diagnostics
- Multiple output formats for different downstream analyses
"""

__author__ = "Jorge Ruballos"
__email__ = "ruballoj@oregonstate.edu"
__date__ = "2025-7-31"
__version__ = "1.0.0"

import pandas as pd
import numpy as np
import os
from datetime import datetime
import sys

# Set matplotlib backend before importing pyplot to avoid Qt issues
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

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
    impute_missing_data,
    plot_imp_diag_histo,  
    track_missingness,
    plot_imp_diag_timeseries, 
    remove_outliers
)

# Try to import VAE imputer (optional)
try:
    from vae_imputer import impute_subject_data 
    VAE_AVAILABLE = True
except ImportError:
    VAE_AVAILABLE = False
    print("WARNING: VAE imputer not available. VAE imputation will be skipped.")

program_name = 'impute_data'

# -------- IMPUTATION SETTINGS --------
imputation_method = 'mean'  # Options: 'mean', 'median', 'forward', 'backward', 'vae'
# VAE variant for imputation (only used if imputation_method = 'vae')
variant = 'encoder + decoder mask'  # Options: 'Zero Imputation', 'Encoder Mask', 'Encoder + Decoder Mask'

output_path = os.path.join(base_path, 'OUTPUT', program_name)
os.makedirs(output_path, exist_ok=True)

# Create diagnostic output directories
sleep_plot_dir = os.path.join(output_path, 'diagnostics', 'sleep_time_plots')
os.makedirs(sleep_plot_dir, exist_ok=True)

print(f"Output directory created at: {output_path}")

# -------- LOGGING --------
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
logfile = os.path.join(output_path, f"imputation_log_{timestamp}.txt")
def log_step(message):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_msg = f"[{ts}] {message}"
    print(full_msg)
    with open(logfile, "a") as f:
        f.write(full_msg + "\n")

# -------- STEP TRACKING FUNCTIONS --------
def track_imputation_step(df, subid, step_name, step_num, tracking_records, feature_cols):
    """Track metrics for each imputation step"""
    n_rows = len(df)
    
    # Count missing values for key columns
    missing_counts = {}
    for col in feature_cols:
        if col in df.columns:
            missing_counts[f'{col}_missing'] = df[col].isna().sum()
            missing_counts[f'{col}_missing_pct'] = round((df[col].isna().sum() / n_rows) * 100, 2) if n_rows > 0 else 0
    
    # Store tracking info
    record = {
        'subid': subid,
        'step_num': step_num,
        'step_name': step_name,
        'n_rows': n_rows,
        'imputation_method': imputation_method,
        **missing_counts
    }
    
    tracking_records.append(record)
    
    # Log summary
    missing_summary = ', '.join([f"{k.replace('_missing', '')}: {v}" for k, v in missing_counts.items() 
                                if k.endswith('_missing') and v > 0])
    log_step(f"Subject {subid} - {step_name}: N={n_rows}, Missing=[{missing_summary}]")
    
    return record

# -------- INPUT DATA PATHS --------
# Path to the unimputed dataset from combine_clean_data pipeline
INPUT_DATA_PATH = os.path.join(base_path, 'OUTPUT', 'combine_clean_data', 'labeled_daily_data_unimputed.csv')

# Validate input file exists
if not os.path.exists(INPUT_DATA_PATH):
    log_step(f"ERROR: Input file not found: {INPUT_DATA_PATH}")
    sys.exit(1)

# -------- FEATURE DEFINITIONS --------
# These should match the features from your original pipeline
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
    'start_sleep_time', 'end_sleep_time' 
]

ACTIVITY_FEATURES = [
    'Night_Bathroom_Visits', 'Night_Kitchen_Visits'
]

CLINICAL_FEATURES = [
    'amyloid'
] 

DEMO_FEATURES = [
    'birthyr', 'sex', 'hispanic', 'race', 'educ', 'livsitua', 'independ', 'residenc', 'alzdis', 'maristat', 'moca_avg', 'cogstat',
    'primlang', 'mocatots', 'age_at_visit', 'age_bucket', 'educ_group', 'moca_category', 'race_group'
]

# Features that will be imputed
IMPUTABLE_FEATURES = FEATURES + EMFIT_FEATURES + ACTIVITY_FEATURES

# -------- LOAD DATA --------
log_step("Loading unimputed dataset...")
input_df = pd.read_csv(INPUT_DATA_PATH)
log_step(f"Loaded dataset with shape: {input_df.shape}")
log_step(f"Unique subjects: {input_df['subid'].nunique()}")

# Convert date column to proper date type
input_df['date'] = pd.to_datetime(input_df['date']).dt.date

# Get list of subjects
subject_ids = sorted(input_df['subid'].unique())
log_step(f"Processing {len(subject_ids)} subjects")

# -------- IMPUTATION PIPELINE --------
imputed_data = []
missingness_records = []
tracking_records = []
raw_data_before_imputation = []

# Create output directory for raw data before imputation
raw_output_path = os.path.join(output_path, 'raw_before_imputation')
os.makedirs(raw_output_path, exist_ok=True)

for subid in subject_ids:
    log_step(f"\n=== Processing Subject {subid} ===")
    
    # Extract subject data
    subject_df = input_df[input_df['subid'] == subid].copy()
    subject_df = subject_df.sort_values('date').reset_index(drop=True)
    
    if subject_df.empty:
        log_step(f"No data found for subject {subid}, skipping...")
        continue
    
    # STEP 1: Track initial state
    available_imputable_features = [col for col in IMPUTABLE_FEATURES if col in subject_df.columns]
    track_imputation_step(subject_df, subid, "initial_load", 1, tracking_records, available_imputable_features)
    
    # Store original data for comparison
    original_subject_df = subject_df.copy()
    
    # Save raw data before imputation
    raw_data_before_imputation.append(subject_df.copy())
    subject_df.to_csv(os.path.join(raw_output_path, f"{subid}_raw_before_imputation.csv"), index=False)
    
    # STEP 2: Compute missingness percentages
    total_days = len(subject_df)
    missing_pct_columns = {}
    
    for feat in available_imputable_features:
        missing_count = subject_df[feat].isna().sum()
        missing_pct = round((missing_count / total_days) * 100, 2) if total_days > 0 else np.nan
        missing_pct_columns[f"{feat}_missing_pct"] = missing_pct
    
    # Add missing percentages to dataframe
    for col_name, value in missing_pct_columns.items():
        subject_df[col_name] = value
    
    # STEP 3: Create missingness mask
    mask_features = [col for col in available_imputable_features if col in subject_df.columns]
    missingness_mask = subject_df[mask_features].isna().astype(int)
    missingness_mask.columns = [col + '_mask' for col in missingness_mask.columns]
    
    track_imputation_step(subject_df, subid, "before_imputation", 2, tracking_records, available_imputable_features)
    
    # Count missing values before imputation
    missing_before = subject_df[available_imputable_features].isna().sum()
    
    # STEP 4: Apply imputation
    if imputation_method == 'vae' and VAE_AVAILABLE:
        log_step(f"Applying VAE imputation with variant: {variant}")
        subject_df = impute_subject_data(subject_df, input_columns=available_imputable_features, 
                                       epochs=30, variant=variant)
    else:
        if imputation_method == 'vae' and not VAE_AVAILABLE:
            log_step("VAE not available, falling back to mean imputation")
            imputation_method_actual = 'mean'
        else:
            imputation_method_actual = imputation_method
        
        log_step(f"Applying {imputation_method_actual} imputation")    
        subject_df = impute_missing_data(subject_df, columns=available_imputable_features, 
                                       method=imputation_method_actual, multivariate=False)
    
    track_imputation_step(subject_df, subid, "after_imputation", 3, tracking_records, available_imputable_features)
    
    # STEP 4: Add missingness mask back to dataframe
    subject_df = pd.concat([subject_df, missingness_mask], axis=1)
    
    track_imputation_step(subject_df, subid, "after_mask_addition", 4, tracking_records, available_imputable_features)
    
    # Count missing values after imputation
    missing_after = subject_df[available_imputable_features].isna().sum()
    
    # STEP 5: Save missingness tracking info
    missingness_record = track_missingness(
        original_df=original_subject_df,
        imputed_df=subject_df,
        columns=available_imputable_features,
        subid=subid,
        imputation_method=imputation_method
    )
    missingness_records.append(missingness_record)
    
    # STEP 6: Generate diagnostic plots if requested
    if GENERATE_HISTO:
        diagnostic_output_path = os.path.join(output_path, 'diagnostics', imputation_method, 'histograms')
        plot_imp_diag_histo(
            original_df=original_subject_df,
            imputed_df=subject_df,
            columns=available_imputable_features,
            subid=subid,
            output_dir=diagnostic_output_path,
            method_name=imputation_method
        )
    
    if GENERATE_LINE:
        diagnostic_output_path = os.path.join(output_path, 'diagnostics', imputation_method, 'lineplots')
        plot_imp_diag_timeseries(
            original_df=original_subject_df,
            imputed_df=subject_df,
            columns=available_imputable_features,
            subid=subid,
            output_dir=diagnostic_output_path,
            method_name=imputation_method
        )
    
    # STEP 7: Create engineered features
    log_step(f"Creating engineered features for subject {subid}")
    engineered_cols = {}
    
    for col in available_imputable_features:
        if col in subject_df.columns and not subject_df[col].isna().all():
            mean = subject_df[col].mean()
            std = subject_df[col].std()
            if std > 0:  # Avoid division by zero
                engineered_cols[col + '_norm'] = (subject_df[col] - mean) / std
            else:
                engineered_cols[col + '_norm'] = subject_df[col] - mean
            
            engineered_cols[col + '_delta'] = subject_df[col] - mean
            engineered_cols[col + '_delta_1d'] = subject_df[col].diff()
            engineered_cols[col + '_ma_7'] = subject_df[col].rolling(window=7, min_periods=1).mean()
    
    # Add engineered features to dataframe
    if engineered_cols:
        subject_df = pd.concat([subject_df, pd.DataFrame(engineered_cols)], axis=1)
    
    track_imputation_step(subject_df, subid, "after_feature_engineering", 7, tracking_records, available_imputable_features)
    
    # STEP 8: Final data preparation
    subject_df['subid'] = subid  # Ensure subid is present
    imputed_data.append(subject_df.copy())
    
    # Print imputation report
    log_step(f"Subject {subid} - Imputation report ({imputation_method}):")
    for feature in available_imputable_features:
        log_step(f"  {feature}: {missing_before[feature]} missing → {missing_after[feature]} missing")

# -------- FINAL DATA ASSEMBLY --------
log_step("Assembling final dataset...")
final_df = pd.concat(imputed_data, ignore_index=True)

# Final quality checks
n_dupes = final_df.duplicated(subset=['subid', 'date']).sum()
if n_dupes > 0:
    log_step(f"WARNING: {n_dupes} duplicate rows found in final dataset!")
    dup_df = final_df[final_df.duplicated(['subid', 'date'], keep=False)]
    dup_df.to_csv(os.path.join(output_path, 'debug_duplicated_subid_date.csv'), index=False)
else:
    log_step("SUCCESS: No duplicate rows in final dataset.")

log_step(f"Final dataset shape: {final_df.shape}")
log_step(f"Final unique subjects: {final_df['subid'].nunique()}")

# -------- COLUMN REORDERING --------
# Create temporal columns list (including engineered features)
temporal_cols = []
for base_col in available_imputable_features:
    if base_col in final_df.columns:
        temporal_cols.append(base_col)
        # Add engineered variants if they exist
        for suffix in ['_norm', '_delta', '_delta_1d', '_ma_7']:
            eng_col = base_col + suffix
            if eng_col in final_df.columns:
                temporal_cols.append(eng_col)

# Safely reorder columns
available_cols = ['subid', 'date']
col_groups = [
    temporal_cols,
    [col for col in CLINICAL_FEATURES if col in final_df.columns],
    [col for col in DEMO_FEATURES if col in final_df.columns],
    [col for col in final_df.columns if col.startswith('days_since_')],
    [col for col in final_df.columns if col.startswith('label_')]
]

for col_group in col_groups:
    available_cols.extend([col for col in col_group if col in final_df.columns and col not in available_cols])

# Reorder final dataframe
final_df = final_df[available_cols]

# -------- SAVE OUTPUTS --------
csv_filename = f"labeled_daily_data_{imputation_method}_imputed.csv"
final_df.to_csv(os.path.join(output_path, csv_filename), index=False)
log_step(f"Saved imputed dataset: {csv_filename} with shape {final_df.shape}")

# Save tracking records
if tracking_records:
    tracking_df = pd.DataFrame(tracking_records)
    tracking_df.to_csv(os.path.join(output_path, f'imputation_step_tracking_{imputation_method}.csv'), index=False)
    log_step("Saved step-by-step imputation tracking data.")

# Save missingness report
if missingness_records:
    missingness_df = pd.DataFrame(missingness_records)
    missingness_report_path = os.path.join(output_path, f"missingness_report_{imputation_method}.csv")
    missingness_df.to_csv(missingness_report_path, index=False)
    log_step(f"Saved missingness report: {missingness_report_path}")

# Save raw data before imputation (concatenated)
if raw_data_before_imputation:
    raw_df = pd.concat(raw_data_before_imputation, ignore_index=True)
    raw_df.to_csv(os.path.join(output_path, 'all_subjects_raw_before_imputation.csv'), index=False)
    log_step("Saved concatenated raw data before imputation.")

# Final summary
final_summary = pd.DataFrame({
    'n_subids': [final_df['subid'].nunique()],
    'total_rows': [len(final_df)],
    'n_features': [len(final_df.columns)],
    'imputation_method': [imputation_method],
    'n_duplicates': [final_df.duplicated(subset=['subid', 'date']).sum()],
    'processing_timestamp': [timestamp]
})
final_summary.to_csv(os.path.join(output_path, f'imputation_summary_{imputation_method}.csv'), index=False)
log_step("Saved final imputation summary.")

log_step(f"\n=== IMPUTATION PIPELINE COMPLETED ===")
log_step(f"Method: {imputation_method}")
log_step(f"Subjects processed: {final_df['subid'].nunique()}")
log_step(f"Final dataset shape: {final_df.shape}")
log_step(f"Output saved to: {output_path}")