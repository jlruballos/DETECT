#!/usr/bin/env python3
"""
EMFIT Sleep Processing Pipeline

This script processes sleep data from EMFIT bed sensors for participants in the DETECT study.
Each day can have multiple sleep periods (e.g., nighttime sleep and naps). The script identifies
and labels the longest sleep as 'sleep_1', and any additional ones as 'sleep_2', 'sleep_3', etc.

Main steps include:
- Loading and cleaning the raw EMFIT data
- Dropping rows with missing start or end times
- Labeling daily sleep periods by length
- Engineering sleep features (e.g., time in bed, sleep duration)
- Saving summaries and cleaned datasets

Outputs include:
- A full labeled dataset (`emfit_sleep_labeled.csv`)
- A filtered dataset with only the longest sleep (`emfit_sleep_1.csv`)
- CSV summaries of missing data and row counts per participant
"""

__author__ = "Jorge Ruballos"
__email__ = "ruballoj@oregonstate.edu"
__date__ = "2025-07-27"
__version__ = "1.0.0"

import pandas as pd
import os
from datetime import datetime
import sys

program_name = 'emfit_processing'

#-----Function to Assign sleep label per subid and date -----
def assign_sleep_label(group):
    group = group.sort_values(by='sleep_period', ascending=False).copy()
    group['sleep_label'] = [ f"sleep_{i+1}" for i in range(len(group))]
    return group

#------Load EMFIT data -----
base_path = '/mnt/d/DETECT'
output_path = os.path.join(base_path, 'OUTPUT', program_name)
os.makedirs(output_path, exist_ok=True)  # Create output directory if it does not exist
#emfit_path = os.path.join(base_path, 'DETECT_Data', 'Emfit_Data', 'summary', 'Emfit_Summary_Data_DETECT_2024-12-16.csv')
emfit_path = os.path.join(base_path, 'DETECT_Data_080825', 'Sensor_Data','Emfit_Data', 'Summary', 'Emfit_Summary_Data.csv')

# Add helpers directory to system path
sys.path.append(os.path.join(base_path, 'HELPERS'))
from helpers import (
    proc_emfit_data  
)

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

df = pd.read_csv(emfit_path, parse_dates=['date', 'start_sleep', 'end_sleep'])

#-----Step 0: Log the start of the EMFIT data loading process -----
log_step("Starting to load EMFIT data.")
#-----Count the number of unique subids -----
num_subids = df['subid'].nunique()
print(f"Count data for {emfit_path}:")
print(f"Number of unique subids: {num_subids}")
log_step(f"Number of unique subids when loading EMFIT data: {num_subids}")

#------count totlal number of rows -----
total_rows = len(df)
print(f"Total number of rows before dropping missing values when loading EMFIT data: {total_rows}")
log_step(f"Total number of rows before dropping missing values when loading EMFIT data: {total_rows}")

#-----Count the number of rows per subid----
row_counts_step_0 = df['subid'].value_counts().sort_index()
row_counts_step_0.to_csv(os.path.join(output_path, 'row_counts_per_subid_step_0.csv'), header=True)
print(f"Number of rows per subid:\n{row_counts_step_0}")
log_step("Loaded EMFIT data and counted rows per subid.")

#------count the number of missing or NaN values in the all feature columns-----
missing_counts_step_0 = df.isnull().sum()
missing_counts_step_0 = missing_counts_step_0[missing_counts_step_0 > 0]
print(f"Missing values in each column:\n{missing_counts_step_0}")
log_step(f"Missing values in each column:\n{missing_counts_step_0}")

#------Step 1: Drop rows with missing start_sleep or end_sleep -----
log_step("Dropping rows with missing start_sleep or end_sleep.")

#------Drop rows with missing start_sleep or end_sleep -----
df = df.dropna(subset=['start_sleep', 'end_sleep'])

#-----log number of unique subids after dropping missing values-----
num_subids_step_1 = df['subid'].nunique()
print(f"Number of unique subids after dropping missing values: {num_subids_step_1}")
log_step(f"Number of unique subids after dropping missing values: {num_subids_step_1}")

#----------count total number of rows after dropping missing values
total_rows_step_1 = len(df)
print(f"Total number of rows after dropping missing values: {total_rows_step_1}")
log_step(f"Total number of rows after dropping missing values: {total_rows_step_1}")

#----Log the number of rows after dropping missing values----
row_counts_step_1 = df['subid'].value_counts().sort_index()
row_counts_step_1.to_csv(os.path.join(output_path, 'row_counts_per_subid_step_1.csv'), header=True)
print(f"Number of rows after dropping rows with missing start_sleep or end_sleep:\n{row_counts_step_1}")

#------Count missing values in each column after dropping missing values-----
missing_counts_step_1 = df.isnull().sum()
missing_counts_step_1 = missing_counts_step_1[missing_counts_step_1 > 0]
print(f"Missing values in each column after dropping missing values:\n{missing_counts_step_1}")
log_step(f"Missing values in each column after dropping missing values:\n{missing_counts_step_1}")

#------Step 2: Assign sleep labels and feature engineering -----
log_step("Assigning sleep labels based on sleep periods and EMFIT data processing (converting time to minutes, feature creation: start_sleep_time, end_sleep_time etc.).")
#------group by subid and date -----
df = df.groupby(['subid', 'date'], group_keys=False).apply(assign_sleep_label)

#-------- EMFIT DATA PREPROCESSING --------
df = proc_emfit_data(df)

#------Drop sin and cos columns used for sleep period calculation -----
sin_cos_cols = [col for col in df.columns if col.endswith('_sin') or col.endswith('_cos')]

if sin_cos_cols:
    df = df.drop(columns=sin_cos_cols)
    print(f"Dropped sin/cos columns: {sin_cos_cols}")
else:
    print("No sin/cos columns found to drop.")

#------Log the number of rows after preprocessing -----
total_rows_step_2 = len(df)
print(f"Total number of rows after preprocessing: {total_rows_step_2}")
log_step(f"Total number of rows after preprocessing: {total_rows_step_2}")

#------Count the number of unique subids after preprocessing -----
num_subids_step_2 = df['subid'].nunique()
print(f"Number of unique subids after preprocessing: {num_subids_step_2}")
log_step(f"Number of unique subids after preprocessing: {num_subids_step_2}")

#------Count the number of rows per subid after preprocessing -----
row_counts_step_2 = df['subid'].value_counts().sort_index()
row_counts_step_2.to_csv(os.path.join(output_path, 'row_counts_per_subid_step_2.csv'), header=True)
print(f"Number of rows per subid after preprocessing:\n{row_counts_step_2}")

#------count missing values in each column after preprocessing -----
missing_counts_step_2 = df.isnull().sum()
missing_counts_step_2 = missing_counts_step_2[missing_counts_step_2 > 0]
print(f"Missing values in each column after preprocessing:\n{missing_counts_step_2}")
log_step(f"Missing values in each column after preprocessing:\n{missing_counts_step_2}")

#------Save the processed data -----
output_path_emfit = os.path.join(output_path, 'emfit_sleep_labeled.csv')
os.makedirs(os.path.dirname(output_path), exist_ok=True)
df.to_csv(output_path_emfit, index=False)

#------Count the numbrer of sleep periods per subid -----
sleep_counts = df.groupby('subid')['sleep_label'].value_counts().unstack(fill_value=0)
print(f"Number of sleep periods per subid:\n{sleep_counts}")
sleep_counts.to_csv(os.path.join(output_path, 'sleep_counts_per_subid.csv'))

#------Step 3: Filter for sleep_1 -----
log_step("Filtering for sleep_1 and saving the data.")

#---filter for sleep_1 and save -----
df_sleep_1 = df[df['sleep_label'] == "sleep_1"]

#---count the number of unique subids for sleep_1 -----
num_subids_step_3 = df_sleep_1['subid'].nunique()
print(f"Number of unique subids for sleep_1: {num_subids_step_3}")
log_step(f"Number of unique subids for sleep_1: {num_subids_step_3}")

#---Log the number of rows for sleep_1 -----
num_rows_step_3 = len(df_sleep_1)
print(f"Number of rows for sleep_1: {num_rows_step_3}")
log_step(f"Number of rows for sleep_1: {num_rows_step_3}")

#---Count the number of rows per subid for sleep_1 -----
row_counts_step_3 = df_sleep_1['subid'].value_counts().sort_index()
row_counts_step_3.to_csv(os.path.join(output_path, 'row_counts_per_subid_step_3.csv'))
print(f"Number of rows per subid for sleep_1:\n{row_counts_step_3}")

#---Count missing values in each column for sleep_1 -----
missing_counts_step_3 = df_sleep_1.isnull().sum()
missing_counts_step_3 = missing_counts_step_3[missing_counts_step_3 > 0]
print(f"Missing values in each column for sleep_1:\n{missing_counts_step_3}")
log_step(f"Missing values in each column for sleep_1:\n{missing_counts_step_3}")

#---Save the filtered data for sleep_1 -----
output_path_sleep_1 = os.path.join(output_path, 'emfit_sleep_1.csv')
df_sleep_1.to_csv(output_path_sleep_1, index=False)

#-----Final frequency table of number of rows per subid through the entire pipeline -----
final_row_counts = pd.DataFrame({
    'step_0_Initial': row_counts_step_0,
    'step_1_Dropped_NAs': row_counts_step_1,
    'step_2_Processed': row_counts_step_2,
    'step_3_Filtered_Sleep_1': row_counts_step_3
})

#-----Add 'included_in_final' column: 1 if subid exists in step 3, else 0-----
final_row_counts['included_in_final'] = final_row_counts['step_3_Filtered_Sleep_1'].notna().astype(int)

final_row_counts.to_csv(os.path.join(output_path, 'final_row_counts_per_subid.csv'))
print(f"Final frequency table of number of rows per subid saved to {os.path.join(output_path, 'final_row_counts_per_subid.csv')}")

#-----Final Summary of the pipeline -----
summary_all = {
    "step_0_initial": {
        "unique_subids": num_subids,
        "n_rows": total_rows
    },
    "step_1_dropped_na": {
        "unique_subids": num_subids_step_1,
        "n_rows": total_rows_step_1
    },
    "step_2_processed": {
        "unique_subids": num_subids_step_2,
        "n_rows": total_rows_step_2
    },
    "step_3_sleep_1": {
        "unique_subids": num_subids_step_3,
        "n_rows": num_rows_step_3
    }
}

summary_df = pd.DataFrame.from_dict(summary_all, orient='index')

summary_df.to_csv(os.path.join(output_path, 'summary_pipeline_overview.csv'))
print("Combined step summary saved to: summary_pipeline_overview.csv")

#-----output a summary fo the missing counts across all steps -----
missing_counts_all = {
    "step_0_initial": missing_counts_step_0,
    "step_1_dropped_na": missing_counts_step_1,
    "step_2_processed": missing_counts_step_2,
    "step_3_sleep_1": missing_counts_step_3
}

# Convert to dataframe and fill missing with 0
missing_counts_df = pd.DataFrame(missing_counts_all).fillna(0).astype(int)

# Save to CSV
missing_summary_path = os.path.join(output_path, 'missing_counts_summary.csv')
missing_counts_df.to_csv(missing_summary_path)
print(f"Missing value summary saved to: {missing_summary_path}")

