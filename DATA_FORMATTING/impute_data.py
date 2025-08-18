#!/usr/bin/env python3
"""
Imputation Pipeline for DETECT Sensor Data (subject- & feature-aware)

This script:
- Loads the unimputed daily dataset from combine_clean_data
- For each SUBJECT and each FEATURE:
    * Only imputes if that subject has at least MIN_REAL_POINTS real (non-NaN) values
    * Skips imputation if the column is 100% missing for that subject (no modality creation)
    * Engineers features ONLY for columns with >= MIN_REAL_POINTS real values
- Adds missingness masks
- Tracks step-by-step missingness and writes logs/CSVs

Author: Jorge Ruballos
Email: ruballoj@oregonstate.edu
Date: 2025-07-31
Version: 1.1.0
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
import sys

# --- Matplotlib setup for optional diagnostics ---
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# -------- CONFIG --------
USE_WSL = True  # True if running inside WSL (Linux on Windows)
GENERATE_HISTO = False
GENERATE_LINE = False

# Minimum number of real points a subject must have for a column to be eligible
MIN_REAL_POINTS = 3  # <-- tune this; 3–7 is a reasonable range

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
imputation_method = 'mean'  # 'mean', 'median', 'forward', 'backward', 'vae'
variant = 'encoder + decoder mask'  # used only if imputation_method == 'vae'

output_path = os.path.join(base_path, 'OUTPUT', program_name)
os.makedirs(output_path, exist_ok=True)

# Create diagnostic output directories
sleep_plot_dir = os.path.join(output_path, 'diagnostics', 'sleep_time_plots')
os.makedirs(sleep_plot_dir, exist_ok=True)

print(f"Output directory: {output_path}")

# -------- LOGGING --------
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
logfile = os.path.join(output_path, f"imputation_log_{timestamp}.txt")
def log_step(message: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_msg = f"[{ts}] {message}"
    print(full_msg)
    with open(logfile, "a") as f:
        f.write(full_msg + "\n")

# -------- STEP TRACKING --------
def track_imputation_step(df, subid, step_name, step_num, tracking_records, feature_cols):
    """Track metrics for each imputation step."""
    n_rows = len(df)
    missing_counts = {}
    for col in feature_cols:
        if col in df.columns:
            miss = df[col].isna().sum()
            missing_counts[f'{col}_missing'] = miss
            missing_counts[f'{col}_missing_pct'] = round((miss / n_rows) * 100, 2) if n_rows > 0 else 0

    record = {
        'subid': subid,
        'step_num': step_num,
        'step_name': step_name,
        'n_rows': n_rows,
        'imputation_method': imputation_method,
        **missing_counts
    }
    tracking_records.append(record)

    missing_summary = ', '.join(
        [f"{k.replace('_missing','')}: {v}" for k, v in missing_counts.items() if k.endswith('_missing') and v > 0]
    )
    log_step(f"Subject {subid} - {step_name}: N={n_rows}, Missing=[{missing_summary}]")
    return record

# -------- INPUT DATA PATH --------
INPUT_DATA_PATH = os.path.join(base_path, 'OUTPUT', 'combine_clean_data', 'labeled_daily_data_unimputed.csv')
if not os.path.exists(INPUT_DATA_PATH):
    log_step(f"ERROR: Input file not found: {INPUT_DATA_PATH}")
    sys.exit(1)

# -------- FEATURE DEFINITIONS --------
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

TRANSITION_FEATURES = [
    'transition_count'
]

CLINICAL_FEATURES = [
    'amyloid'
]

DEMO_FEATURES = [
    'birthyr', 'sex', 'hispanic', 'race', 'educ', 'livsitua', 'independ', 'residenc',
    'alzdis', 'maristat', 'moca_avg', 'cogstat', 'primlang', 'mocatots', 'age_at_visit',
    'age_bucket', 'educ_group', 'moca_category', 'race_group', 'maristat_recoded'
]

IMPUTABLE_FEATURES = FEATURES + EMFIT_FEATURES + ACTIVITY_FEATURES + TRANSITION_FEATURES

# -------- LOAD DATA --------
log_step("Loading unimputed dataset...")
input_df = pd.read_csv(INPUT_DATA_PATH)
log_step(f"Loaded dataset with shape: {input_df.shape}")

# Ensure date is date-type
input_df['date'] = pd.to_datetime(input_df['date']).dt.date

unique_subs = input_df['subid'].nunique()
log_step(f"Unique subjects: {unique_subs}")

subject_ids = sorted(input_df['subid'].unique())
log_step(f"Processing {len(subject_ids)} subjects")

# -------- HELPERS --------
def split_columns_for_subject(df: pd.DataFrame, candidates, min_points: int = 1):
    """
    Categorize columns by availability within a subject's dataframe.
      - cols_present: column exists in df
      - cols_with_any_data: >= min_points non-NaN
      - cols_with_missing: subset of above that still have some NaNs
      - cols_all_missing: present but 0 non-NaN
    """
    cols_present = [c for c in candidates if c in df.columns]
    non_na_counts = {c: df[c].notna().sum() for c in cols_present}
    cols_with_any_data = [c for c in cols_present if non_na_counts[c] >= min_points]
    cols_with_missing = [c for c in cols_with_any_data if df[c].isna().any()]
    cols_all_missing = [c for c in cols_present if non_na_counts[c] == 0]
    return cols_present, cols_with_any_data, cols_with_missing, cols_all_missing

# -------- MAIN LOOP --------
imputed_data = []
missingness_records = []
tracking_records = []
raw_data_before_imputation = []

raw_output_path = os.path.join(output_path, 'raw_before_imputation')
os.makedirs(raw_output_path, exist_ok=True)

for subid in subject_ids:
    log_step(f"\n=== Processing Subject {subid} ===")
    subject_df = input_df[input_df['subid'] == subid].copy()
    subject_df = subject_df.sort_values('date').reset_index(drop=True)

    if subject_df.empty:
        log_step(f"No data for subject {subid}; skipping.")
        continue

    # STEP 1: Availability split with MIN_REAL_POINTS guard
    all_imputable_candidates = [c for c in IMPUTABLE_FEATURES if c in subject_df.columns]
    (cols_present,
     cols_with_any_data,
     cols_with_missing,
     cols_all_missing) = split_columns_for_subject(subject_df, all_imputable_candidates, min_points=MIN_REAL_POINTS)

    track_imputation_step(subject_df, subid, "initial_load", 1, tracking_records, cols_present)
    log_step(
        f"Subject {subid} | present:{len(cols_present)} any_data:{len(cols_with_any_data)} "
        f"missing:{len(cols_with_missing)} all_missing:{len(cols_all_missing)} "
        f"(MIN_REAL_POINTS={MIN_REAL_POINTS})"
    )
    if cols_all_missing:
        log_step(f"Subject {subid} - All-missing columns (skipping impute & features): {cols_all_missing}")

    # Save raw before imputation
    original_subject_df = subject_df.copy()
    raw_data_before_imputation.append(subject_df.copy())
    subject_df.to_csv(os.path.join(raw_output_path, f"{subid}_raw_before_imputation.csv"), index=False)

    # STEP 2: Missingness % (optional auditing)
    total_days = len(subject_df)
    for feat in cols_present:
        miss_pct = round((subject_df[feat].isna().sum() / total_days) * 100, 2) if total_days > 0 else np.nan
        subject_df[f"{feat}_missing_pct"] = miss_pct

    # STEP 3: Masks for present columns
    mask_features = cols_present
    missingness_mask = subject_df[mask_features].isna().astype(int)
    missingness_mask.columns = [c + '_mask' for c in missingness_mask.columns]

    track_imputation_step(subject_df, subid, "before_imputation", 2, tracking_records, cols_present)
    missing_before = subject_df[cols_present].isna().sum()

    # STEP 4: Impute ONLY columns that (a) have >= MIN_REAL_POINTS real values AND (b) have some NaNs
    if cols_with_missing:
        if imputation_method == 'vae' and VAE_AVAILABLE:
            log_step(f"Applying VAE imputation ({variant}) to {len(cols_with_missing)} columns")
            subject_df = impute_subject_data(
                subject_df, input_columns=cols_with_missing, epochs=30, variant=variant
            )
        else:
            if imputation_method == 'vae' and not VAE_AVAILABLE:
                log_step("VAE unavailable; falling back to mean imputation")
                imputation_method_actual = 'mean'
            else:
                imputation_method_actual = imputation_method
            log_step(f"Applying {imputation_method_actual} imputation to columns: {cols_with_missing}")
            subject_df = impute_missing_data(
                subject_df, columns=cols_with_missing, method=imputation_method_actual, multivariate=False
            )
    else:
        log_step("No columns require imputation under MIN_REAL_POINTS rule. Skipping imputation.")

    track_imputation_step(subject_df, subid, "after_imputation", 3, tracking_records, cols_present)

    # Add mask back
    subject_df = pd.concat([subject_df, missingness_mask], axis=1)
    track_imputation_step(subject_df, subid, "after_mask_addition", 4, tracking_records, cols_present)

    missing_after = subject_df[cols_present].isna().sum()

    # STEP 5: Save missingness tracking for subject
    missingness_record = track_missingness(
        original_df=original_subject_df,
        imputed_df=subject_df,
        columns=cols_present,           # track only columns actually present for subject
        subid=subid,
        imputation_method=imputation_method
    )
    missingness_records.append(missingness_record)

    # STEP 6: Diagnostics (optional)
    if GENERATE_HISTO:
        diagnostic_output_path = os.path.join(output_path, 'diagnostics', imputation_method, 'histograms')
        plot_imp_diag_histo(
            original_df=original_subject_df,
            imputed_df=subject_df,
            columns=cols_present,
            subid=subid,
            output_dir=diagnostic_output_path,
            method_name=imputation_method
        )

    if GENERATE_LINE:
        diagnostic_output_path = os.path.join(output_path, 'diagnostics', imputation_method, 'lineplots')
        plot_imp_diag_timeseries(
            original_df=original_subject_df,
            imputed_df=subject_df,
            columns=cols_present,
            subid=subid,
            output_dir=diagnostic_output_path,
            method_name=imputation_method
        )

    # STEP 7: Engineer features ONLY for columns that had >= MIN_REAL_POINTS real values (honest features)
    log_step(f"Creating engineered features for subject {subid}")
    engineered_cols = {}
    for col in cols_with_any_data:
        if col not in subject_df.columns:
            continue
        series = subject_df[col]
        # If still all NaN (edge case), skip
        if series.notna().sum() == 0:
            continue

        mean = series.mean(skipna=True)
        std = series.std(skipna=True)

        # Normalized
        if pd.notna(std) and std > 0:
            engineered_cols[col + '_norm'] = (series - mean) / std
        else:
            engineered_cols[col + '_norm'] = series - mean

        # Deltas and rolling mean
        engineered_cols[col + '_delta'] = series - mean
        engineered_cols[col + '_delta_1d'] = series.diff()
        engineered_cols[col + '_ma_7'] = series.rolling(window=7, min_periods=1).mean()

    if engineered_cols:
        subject_df = pd.concat([subject_df, pd.DataFrame(engineered_cols)], axis=1)

    track_imputation_step(subject_df, subid, "after_feature_engineering", 7, tracking_records, cols_present)

    # STEP 8: Finalize subject
    subject_df['subid'] = subid
    imputed_data.append(subject_df.copy())

    # Report per-feature changes
    log_step(f"Subject {subid} - Imputation report ({imputation_method}):")
    for feature in cols_present:
        before = int(missing_before.get(feature, 0))
        after = int(missing_after.get(feature, 0))
        log_step(f"  {feature}: {before} missing → {after} missing")

# -------- FINAL DATA ASSEMBLY --------
log_step("Assembling final dataset...")
final_df = pd.concat(imputed_data, ignore_index=True)

# Duplicates check
n_dupes = final_df.duplicated(subset=['subid', 'date']).sum()
if n_dupes > 0:
    log_step(f"WARNING: {n_dupes} duplicate rows found in final dataset!")
    dup_df = final_df[final_df.duplicated(['subid', 'date'], keep=False)]
    dup_df.to_csv(os.path.join(output_path, 'debug_duplicated_subid_date.csv'), index=False)
else:
    log_step("SUCCESS: No duplicate rows in final dataset.")

log_step(f"Final dataset shape: {final_df.shape}")
log_step(f"Final unique subjects: {final_df['subid'].nunique()}")

# -------- COLUMN REORDERING (safe, based on final_df) --------
base_temporal_bases = [c for c in IMPUTABLE_FEATURES if c in final_df.columns]
temporal_cols = []
for base_col in base_temporal_bases:
    if base_col in final_df.columns:
        temporal_cols.append(base_col)
        for suffix in ['_norm', '_delta', '_delta_1d', '_ma_7']:
            eng_col = base_col + suffix
            if eng_col in final_df.columns:
                temporal_cols.append(eng_col)

available_cols = ['subid', 'date']
col_groups = [
    temporal_cols,
    [c for c in CLINICAL_FEATURES if c in final_df.columns],
    [c for c in DEMO_FEATURES if c in final_df.columns],
    [c for c in final_df.columns if c.startswith('days_since_')],
    [c for c in final_df.columns if c.startswith('label_')],
    [c for c in final_df.columns if c.endswith('_mask')],
]

for group in col_groups:
    for col in group:
        if col in final_df.columns and col not in available_cols:
            available_cols.append(col)

final_df = final_df[available_cols]

# -------- SAVE OUTPUTS --------
csv_filename = f"labeled_daily_data_{imputation_method}_imputed_min{MIN_REAL_POINTS}.csv"
final_df.to_csv(os.path.join(output_path, csv_filename), index=False)
log_step(f"Saved imputed dataset: {csv_filename} with shape {final_df.shape}")

# Step-by-step tracking
if tracking_records:
    tracking_df = pd.DataFrame(tracking_records)
    tracking_df.to_csv(os.path.join(output_path, f'imputation_step_tracking_{imputation_method}_min{MIN_REAL_POINTS}.csv'), index=False)
    log_step("Saved step-by-step imputation tracking data.")

# Missingness report
if missingness_records:
    missingness_df = pd.DataFrame(missingness_records)
    missingness_report_path = os.path.join(output_path, f"missingness_report_{imputation_method}_min{MIN_REAL_POINTS}.csv")
    missingness_df.to_csv(missingness_report_path, index=False)
    log_step(f"Saved missingness report: {missingness_report_path}")

# Raw pre-imputation (concatenated)
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
    'min_real_points': [MIN_REAL_POINTS],
    'n_duplicates': [final_df.duplicated(subset=['subid', 'date']).sum()],
    'processing_timestamp': [timestamp]
})
final_summary.to_csv(os.path.join(output_path, f'imputation_summary_{imputation_method}_min{MIN_REAL_POINTS}.csv'), index=False)
log_step("Saved final imputation summary.")

log_step("\n=== IMPUTATION PIPELINE COMPLETED ===")
log_step(f"Method: {imputation_method}")
log_step(f"MIN_REAL_POINTS: {MIN_REAL_POINTS}")
log_step(f"Subjects processed: {final_df['subid'].nunique()}")
log_step(f"Final dataset shape: {final_df.shape}")
log_step(f"Output saved to: {output_path}")
