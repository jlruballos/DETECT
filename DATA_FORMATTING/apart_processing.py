#!/usr/bin/env python3
"""
Time Apart Data Processing Pipeline

Cleans daily "time apart" hours from in-home sensors for DETECT participants.
Expected columns: subid, homeid, Date, Apart_Proxy_Hours, ExactlyOne_Hours.

Main steps:
- Load dataset and log counts
- Drop negative values (Apart_Proxy_Hours, ExactlyOne_Hours)
- Drop rows missing Date, subid, Apart_Proxy_Hours, or ExactlyOne_Hours
- Parse 'Date' to calendar date
- Check and drop duplicate (subid, Date) rows
- Save cleaned file and step-by-step summaries

Outputs:
- time_apart_cleaned.csv
- pipeline_log_YYYYMMDD_HHMMSS.txt
- row_counts_step_*.csv, missing_counts_step_*.csv
- summary_pipeline_overview.csv, missing_counts_summary.csv
- duplicate_entries.csv (if any)
"""

__author__ = "Jorge Ruballos"
__email__ = "ruballoj@oregonstate.edu"
__date__ = "2025-08-17"
__version__ = "1.0.0"

import pandas as pd
import os
from datetime import datetime
import sys

program_name = 'apart_processing'
base_path = '/mnt/d/DETECT'
output_path = os.path.join(base_path, 'OUTPUT', program_name)
os.makedirs(output_path, exist_ok=True)

activity_path = os.path.join(base_path, 'OUTPUT', 'time_apart', 'apart_proxy_hours_by_day.csv')

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
log_step("Step 0: Loading Time Apart data.")
df = pd.read_csv(activity_path)
df['subid'] = df['subid'].astype(str)

num_subids_0 = df['subid'].nunique()
num_homeids_0 = df['homeid'].nunique()
total_rows_0 = len(df)
log_step(f"Step 0: {num_subids_0} unique subids, {num_homeids_0} unique homeids, {total_rows_0} total rows.")

row_counts_0 = df['subid'].value_counts().sort_index()
row_counts_0.to_csv(os.path.join(output_path, 'row_counts_step_0.csv'))

missing_0 = df.isnull().sum()
missing_0[missing_0 > 0].to_csv(os.path.join(output_path, 'missing_counts_step_0.csv'))

#------ Step 1: Drop Negative Apart_Proxy_Hours and ExactlyOne_Hours ------
log_step("Step 1: Dropping rows with negative 'Apart_Proxy_Hours' and 'ExactlyOne_Hours' values.")
if 'Apart_Proxy_Hours' in df.columns:
    df = df[df['Apart_Proxy_Hours'] >= 0]
if 'ExactlyOne_Hours' in df.columns:
    df = df[df['ExactlyOne_Hours'] >= 0]

num_subids_1 = df['subid'].nunique()
num_homeids_1 = df['homeid'].nunique()
total_rows_1 = len(df)
log_step(f"Step 1: {num_subids_1} unique subids, {num_homeids_1} unique homeids, {total_rows_1} total rows after removing negatives.")

row_counts_1 = df['subid'].value_counts().sort_index()
row_counts_1.to_csv(os.path.join(output_path, 'row_counts_step_1.csv'))

missing_1 = df.isnull().sum()
missing_1[missing_1 > 0].to_csv(os.path.join(output_path, 'missing_counts_step_1.csv'))

#------ Step 2: Drop rows with missing 'Date' or 'subid' or 'Apart_Proxy_Hours' or 'ExactlyOne_Hours' ------
log_step("Step 2: Dropping rows with missing 'Date' or 'subid' or 'Apart_Proxy_Hours' or 'ExactlyOne_Hours'.")
df = df.dropna(subset=['Date', 'subid', 'Apart_Proxy_Hours', 'ExactlyOne_Hours'])

num_subids_2 = df['subid'].nunique()
num_homeids_2 = df['homeid'].nunique()
total_rows_2 = len(df)
log_step(f"Step 2: {num_subids_2} unique subids, {num_homeids_2} unique homeids, {total_rows_2} total rows after dropping NAs.")

row_counts_2 = df['subid'].value_counts().sort_index()
row_counts_2.to_csv(os.path.join(output_path, 'row_counts_step_2.csv'))

missing_2 = df.isnull().sum()
missing_2[missing_2 > 0].to_csv(os.path.join(output_path, 'missing_counts_step_2.csv'))

#------ Step 3: Convert 'Date' to datetime.date ------
log_step("Step 3: Converting 'Date' column to datetime.date.")
df['Date'] = pd.to_datetime(df['Date'], errors='coerce').dt.date
df = df.dropna(subset=['Date'])  # In case of parse failure

num_subids_3 = df['subid'].nunique()
num_homeids_3 = df['homeid'].nunique()
total_rows_3 = len(df)
log_step(f"Step 3: {num_subids_3} unique subids, {num_homeids_3} unique homeids, {total_rows_3} total rows after date parsing.")

row_counts_3 = df['subid'].value_counts().sort_index()
row_counts_3.to_csv(os.path.join(output_path, 'row_counts_step_3.csv'))

missing_3 = df.isnull().sum()
missing_3[missing_3 > 0].to_csv(os.path.join(output_path, 'missing_counts_step_3.csv'))

#------ Step 4: Check for duplicate (subid, Date) ------
log_step("Step 4: Checking for duplicate (subid, Date) entries.")
duplicate_mask = df.duplicated(subset=['subid', 'Date'], keep=False)
duplicates = df[duplicate_mask]
if not duplicates.empty:
    duplicate_path = os.path.join(output_path, 'duplicate_entries.csv')
    duplicates.to_csv(duplicate_path, index=False)
    log_step(f"Step 4: Found {len(duplicates)} duplicate rows. Saved to {duplicate_path}")
    log_step("Dropping duplicates.")
    #-----Drop duplicates-----
    df = df.drop_duplicates(subset=['subid', 'Date'])
else:
    log_step("Step 4: No duplicate (subid, Date) entries found.")

num_subids_4 = df['subid'].nunique()
num_homeids_4 = df['homeid'].nunique()
total_rows_4 = len(df)
log_step(f"Step 4: {num_subids_4} unique subids, {num_homeids_4} unique homeids, {total_rows_4} total rows after duplicate check.")

row_counts_4 = df['subid'].value_counts().sort_index()
row_counts_4.to_csv(os.path.join(output_path, 'row_counts_step_4.csv'))

missing_4 = df.isnull().sum()
missing_4[missing_4 > 0].to_csv(os.path.join(output_path, 'missing_counts_step_4.csv'))

#------ Save Cleaned Data ------
output_cleaned = os.path.join(output_path, 'time_apart_cleaned.csv')
df.to_csv(output_cleaned, index=False)
log_step(f"Final cleaned data saved to {output_cleaned}")

#------ Final Summary Table ------
summary = pd.DataFrame({
    'unique_subids': [num_subids_0, num_subids_1, num_subids_2, num_subids_3, num_subids_4],
    'unique_homeids': [num_homeids_0, num_homeids_1, num_homeids_2, num_homeids_3, num_homeids_4],
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
