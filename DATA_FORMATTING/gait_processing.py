#!/usr/bin/env python3
"""
Gait Speed Processing Pipeline

This script processes daily gait speed data from in-home NYCE sensors for participants in the DETECT study.
It merges the gait data with subject IDs, filters and summarizes the data, and outputs cleaned files.

Main steps include:
- Loading and validating the gait dataset
- Merging with subject mapping file to assign `subid`
- Converting timestamps to dates
- Dropping rows with missing, negative, or duplicate values
- Aggregating daily mean gait speed per subject (at the end)
- Filtering out outliers above 300 cm/s
- Saving cleaned datasets and summary files

Outputs include:
- A cleaned gait dataset (`gait_speed_cleaned.csv`)
- Summary CSVs of row counts and missingness
"""

__author__ = "Jorge Ruballos"
__email__ = "ruballoj@oregonstate.edu"
__date__ = "2025-07-28"
__version__ = "1.3.0"

import pandas as pd
import os
from datetime import datetime
import sys

program_name = 'gait_speed_processing'

#------ Set up paths ------
base_path = '/mnt/d/DETECT'
output_path = os.path.join(base_path, 'OUTPUT', program_name)
os.makedirs(output_path, exist_ok=True)

#GAIT_PATH = os.path.join(base_path, 'DETECT_Data', 'NYCE_Data', 'COMBINED_NYCE_Area_Data_DETECT_GAIT_Summary.csv')
GAIT_PATH = os.path.join(base_path, 'OUTPUT', 'GAIT', 'COMBINED_NYCE_Area_Data_DETECT_GAIT_Summary.csv')
MAPPING_PATH = os.path.join(base_path, 'DETECT_Data', '_CONTEXT_FILES', 'Study_Home-Subject_Dates_2024-12-16', 'homeids_subids_NYCE.csv')

#------ Logging ------
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
logfile = os.path.join(output_path, f"pipeline_log_{timestamp}.txt")
def log_step(message):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_msg = f"[{ts}] {message}"
    print(full_msg)
    with open(logfile, "a") as f:
        f.write(full_msg + "\n")

#------ Step 0: Load data ------
log_step("Loading gait speed data and mapping file.")
gait_df = pd.read_csv(GAIT_PATH)
mapping_df = pd.read_csv(MAPPING_PATH)

# Filter to ensure each home_id maps to only one subid
mapping_data = mapping_df.groupby('home_id').filter(lambda x: len(x) == 1)
mapping_data = mapping_data.rename(columns={'sub_id': 'subid'})

# Merge gait data with mapping to get subid
log_step("Merging gait data with subject mapping file.")
gait_df = pd.merge(gait_df, mapping_data, left_on='homeid', right_on='home_id', how='inner')

# Count before cleaning
num_subids_0 = gait_df['subid'].nunique()
total_rows_0 = len(gait_df)
log_step(f"Step 0: {num_subids_0} unique subids, {total_rows_0} total rows after merge.")
row_counts_0 = gait_df['subid'].value_counts().sort_index()
row_counts_0.to_csv(os.path.join(output_path, 'row_counts_step_0.csv'))
missing_0 = gait_df.isnull().sum()
missing_0[missing_0 > 0].to_csv(os.path.join(output_path, 'missing_counts_step_0.csv'))

#------ Step 1: Extract date ------
log_step("Extracting date from start_time.")
gait_df['date'] = pd.to_datetime(gait_df['start_time']).dt.date

num_subids_1 = gait_df['subid'].nunique()
total_rows_1 = len(gait_df)
log_step(f"Step 1: {num_subids_1} unique subids, {total_rows_1} total rows after date extraction.")
row_counts_1 = gait_df['subid'].value_counts().sort_index()
row_counts_1.to_csv(os.path.join(output_path, 'row_counts_step_1.csv'))
missing_1 = gait_df.isnull().sum()
missing_1[missing_1 > 0].to_csv(os.path.join(output_path, 'missing_counts_step_1.csv'))

#------ Step 2: Drop missing values ------
log_step("Dropping rows with missing values in 'subid', 'date', or 'gait_speed'.")
gait_df = gait_df.dropna(subset=['subid', 'date', 'gait_speed'])

num_subids_2 = gait_df['subid'].nunique()
total_rows_2 = len(gait_df)
log_step(f"Step 2: {num_subids_2} unique subids, {total_rows_2} total rows after dropping NAs.")
row_counts_2 = gait_df['subid'].value_counts().sort_index()
row_counts_2.to_csv(os.path.join(output_path, 'row_counts_step_2.csv'))
missing_2 = gait_df.isnull().sum()
missing_2[missing_2 > 0].to_csv(os.path.join(output_path, 'missing_counts_step_2.csv'))

#------ Step 3: Drop negative gait speeds ------
log_step("Dropping rows with negative gait speed values.")
neg_count = (gait_df['gait_speed'] < 0).sum()
gait_df = gait_df[gait_df['gait_speed'] >= 0]

num_subids_3 = gait_df['subid'].nunique()
total_rows_3 = len(gait_df)
log_step(f"Dropped {neg_count} rows with negative gait speed. Remaining rows: {total_rows_3}")
row_counts_3 = gait_df['subid'].value_counts().sort_index()
row_counts_3.to_csv(os.path.join(output_path, 'row_counts_step_3.csv'))
missing_3 = gait_df.isnull().sum()
missing_3[missing_3 > 0].to_csv(os.path.join(output_path, 'missing_counts_step_3.csv'))

#------ Step 4: Aggregate gait speed after cleaning ------
log_step("Aggregating daily mean gait speed after cleaning.")
gait_df = gait_df.groupby(['subid', 'date'])['gait_speed'].mean().reset_index()

num_subids_4 = gait_df['subid'].nunique()
total_rows_4 = len(gait_df)
log_step(f"Step 4: {num_subids_4} unique subids, {total_rows_4} total rows after aggregation.")
row_counts_4 = gait_df['subid'].value_counts().sort_index()
row_counts_4.to_csv(os.path.join(output_path, 'row_counts_step_4.csv'))
missing_4 = gait_df.isnull().sum()
missing_4[missing_4 > 0].to_csv(os.path.join(output_path, 'missing_counts_step_4.csv'))

#------ Step 5: Check for duplicates ------
log_step("Checking for duplicate (subid, date) rows.")
duplicates = gait_df.duplicated(subset=['subid', 'date'], keep=False)
dup_rows = gait_df[duplicates]
if not dup_rows.empty:
    log_step(f"Found {len(dup_rows)} duplicate rows. Saving to 'duplicate_entries.csv'.")
    dup_rows.to_csv(os.path.join(output_path, 'duplicate_entries.csv'), index=False)
else:
    log_step("No duplicate rows found.")

#------ Step 6: Drop outliers above xxx cm/s ------
log_step("Dropping rows with gait speed greater than 200 cm/s.")
outlier_count = (gait_df['gait_speed'] > 200).sum()
gait_df = gait_df[gait_df['gait_speed'] <= 200]

num_subids_6 = gait_df['subid'].nunique()
total_rows_6 = len(gait_df)
log_step(f"Dropped {outlier_count} outlier rows. Remaining rows: {total_rows_6}")
row_counts_6 = gait_df['subid'].value_counts().sort_index()
row_counts_6.to_csv(os.path.join(output_path, 'row_counts_step_6.csv'))
missing_6 = gait_df.isnull().sum()
missing_6[missing_6 > 0].to_csv(os.path.join(output_path, 'missing_counts_step_6.csv'))

#------ Save final cleaned data ------
output_final = os.path.join(output_path, 'gait_speed_cleaned.csv')
gait_df.to_csv(output_final, index=False)
log_step(f"Cleaned gait speed data saved to: {output_final}")

#------ Final Summary Table ------
summary = pd.DataFrame({
    'unique_subids': [num_subids_0, num_subids_1, num_subids_2, num_subids_3, num_subids_4, num_subids_6],
    'n_rows':        [total_rows_0, total_rows_1, total_rows_2, total_rows_3, total_rows_4, total_rows_6]
}, index=['step_0_merged', 'step_1_extracted_date', 'step_2_drop_na', 'step_3_no_negatives', 'step_4_aggregated', 'step_6_filter_outliers'])

summary.to_csv(os.path.join(output_path, 'summary_pipeline_overview.csv'))
print("Summary saved to summary_pipeline_overview.csv")

#------ Final Missingness Summary ------
missing_all = pd.DataFrame({
    'step_0': missing_0,
    'step_1': missing_1,
    'step_2': missing_2,
    'step_3': missing_3,
    'step_4': missing_4,
    'step_6': missing_6
}).fillna(0).astype(int)

missing_all.to_csv(os.path.join(output_path, 'missing_counts_summary.csv'))
print("Missing value summary saved to missing_counts_summary.csv")
