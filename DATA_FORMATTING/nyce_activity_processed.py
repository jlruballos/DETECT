#!/usr/bin/env python3
"""
Activity Data Processing Pipeline

This script processes daily activity data from in-home NYCE sensors for participants in the DETECT study.
It performs data cleaning, preprocessing, and missing value summaries. Negative activity values are removed.

Main steps include:
- Loading and validating the activity dataset
- Dropping rows with negative or missing values
- Converting 'Date' to datetime.date
- Checking and reporting duplicate (subid, date) entries
- Saving summary files and cleaned outputs
"""

__author__ = "Jorge Ruballos"
__email__ = "ruballoj@oregonstate.edu"
__date__ = "2025-07-31"
__version__ = "1.0.0"

import pandas as pd
import os
from datetime import datetime
import sys

program_name = 'activity_processing'
base_path = '/mnt/d/DETECT'
output_path = os.path.join(base_path, 'OUTPUT', program_name)
os.makedirs(output_path, exist_ok=True)

activity_path = os.path.join(base_path, 'OUTPUT', 'NYCE_activity', 'daily_area_hours_combined_output.csv')

#------ Logging Setup ------
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
logfile = os.path.join(output_path, f"pipeline_log_{timestamp}.txt")
def log_step(message):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_msg = f"[{ts}] {message}"
    print(full_msg)
    with open(logfile, "a") as f:
        f.write(full_msg + "\n")

#------ Step 0: Load Data ------
log_step("Step 0: Loading Activity data.")
df = pd.read_csv(activity_path)
df['subid'] = df['subid'].astype(str)

num_subids_0 = df['subid'].nunique()
total_rows_0 = len(df)
log_step(f"Step 0: {num_subids_0} unique subids, {total_rows_0} total rows.")

row_counts_0 = df['subid'].value_counts().sort_index()
row_counts_0.to_csv(os.path.join(output_path, 'row_counts_step_0.csv'))

missing_0 = df.isnull().sum()
missing_0[missing_0 > 0].to_csv(os.path.join(output_path, 'missing_counts_step_0.csv'))

#------ Step 1: Drop Negative total_hours ------
log_step("Step 1: Dropping rows with negative 'total_hours' values.")
if 'total_hours' in df.columns:
    df = df[df['total_hours'] >= 0]

num_subids_1 = df['subid'].nunique()
total_rows_1 = len(df)
log_step(f"Step 1: {num_subids_1} unique subids, {total_rows_1} total rows after removing negatives.")

row_counts_1 = df['subid'].value_counts().sort_index()
row_counts_1.to_csv(os.path.join(output_path, 'row_counts_step_1.csv'))

missing_1 = df.isnull().sum()
missing_1[missing_1 > 0].to_csv(os.path.join(output_path, 'missing_counts_step_1.csv'))

#------ Step 2: Drop rows with missing 'Date' or 'subid' ------
log_step("Step 2: Dropping rows with missing 'Date' or 'subid'.")
df = df.dropna(subset=['Date', 'subid'])

num_subids_2 = df['subid'].nunique()
total_rows_2 = len(df)
log_step(f"Step 2: {num_subids_2} unique subids, {total_rows_2} total rows after dropping NAs.")

row_counts_2 = df['subid'].value_counts().sort_index()
row_counts_2.to_csv(os.path.join(output_path, 'row_counts_step_2.csv'))

missing_2 = df.isnull().sum()
missing_2[missing_2 > 0].to_csv(os.path.join(output_path, 'missing_counts_step_2.csv'))

#------ Step 3: Convert 'Date' to datetime.date ------
log_step("Step 3: Converting 'Date' column to datetime.date.")
df['date'] = pd.to_datetime(df['Date'], errors='coerce').dt.date
df = df.dropna(subset=['date'])  # In case of parse failure

num_subids_3 = df['subid'].nunique()
total_rows_3 = len(df)
log_step(f"Step 3: {num_subids_3} unique subids, {total_rows_3} total rows after date parsing.")

row_counts_3 = df['subid'].value_counts().sort_index()
row_counts_3.to_csv(os.path.join(output_path, 'row_counts_step_3.csv'))

missing_3 = df.isnull().sum()
missing_3[missing_3 > 0].to_csv(os.path.join(output_path, 'missing_counts_step_3.csv'))

#------ Step 4: Check for duplicate (subid, date) ------
log_step("Step 4: Checking for duplicate (subid, date) entries.")
duplicate_mask = df.duplicated(subset=['subid', 'date'], keep=False)
duplicates = df[duplicate_mask]
if not duplicates.empty:
    duplicate_path = os.path.join(output_path, 'duplicate_entries.csv')
    duplicates.to_csv(duplicate_path, index=False)
    log_step(f"Step 4: Found {len(duplicates)} duplicate rows. Saved to {duplicate_path}")
    log_step("Dropping duplicates.")
    #-----Drop duplicates-----
    df = df.drop_duplicates(subset=['subid', 'date'])
else:
    log_step("Step 4: No duplicate (subid, date) entries found.")

num_subids_4 = df['subid'].nunique()
total_rows_4 = len(df)
log_step(f"Step 4: {num_subids_4} unique subids, {total_rows_4} total rows after duplicate check.")

row_counts_4 = df['subid'].value_counts().sort_index()
row_counts_4.to_csv(os.path.join(output_path, 'row_counts_step_4.csv'))

missing_4 = df.isnull().sum()
missing_4[missing_4 > 0].to_csv(os.path.join(output_path, 'missing_counts_step_4.csv'))

#------ Save Cleaned Data ------
output_cleaned = os.path.join(output_path, 'activity_cleaned.csv')
df.to_csv(output_cleaned, index=False)
log_step(f"Final cleaned data saved to {output_cleaned}")

#------ Final Summary Table ------
summary = pd.DataFrame({
    'unique_subids': [num_subids_0, num_subids_1, num_subids_2, num_subids_3, num_subids_4],
    'n_rows':        [total_rows_0, total_rows_1, total_rows_2, total_rows_3, total_rows_4]
}, index=['step_0_loaded', 'step_1_no_negatives', 'step_2_drop_na', 'step_3_preprocessed', 'step_4_duplicate'])

summary.to_csv(os.path.join(output_path, 'summary_pipeline_overview.csv'))
print("Summary saved to summary_pipeline_overview.csv")

#------ Final Missingness Summary ------
missing_all = pd.DataFrame({
    'step_0': missing_0,
    'step_1': missing_1,
    'step_2': missing_2,
    'step_3': missing_3,
    'step_4': missing_4
}).fillna(0).astype(int)

missing_all.to_csv(os.path.join(output_path, 'missing_counts_summary.csv'))
print("Missing value summary saved to missing_counts_summary.csv")
