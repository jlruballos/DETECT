#!/usr/bin/env python3
"""
Survey Processing Pipeline

This script processes weekly survey data from the DETECT study to identify event occurrences 
such as falls, hospital visits, accidents, medication changes, and mood changes (blue/lonely).

It performs the following steps:
- Loads raw HUF survey data
- Converts multiple date columns to datetime format
- Fills missing fall dates for confirmed events
- Combines multi-column events into list-format (e.g., hospital dates, accident dates)
- Extracts weekly event frequencies for each event type
- Saves cleaned data, frequency tables, and weekly event plots
- Checks for duplicates and logs all processing steps

Outputs include:
- Cleaned survey dataset (`survey_cleaned.csv`)
- Weekly event frequency CSVs (`event_frequency/`)
- Weekly time-series bar plots per event
- Summary tables of row counts and missingness
- Logged processing steps
"""

__author__ = "Jorge Ruballos"
__email__ = "ruballoj@oregonstate.edu"
__date__ = "2025-07-28"
__version__ = "1.3.0"

import pandas as pd
import os
from datetime import datetime
import sys
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving plots
import matplotlib.pyplot as plt
import seaborn as sns
import json
import numpy as np

program_name = 'survey_processing'

#------ Set up paths ------
base_path = '/mnt/d/DETECT'
output_path = os.path.join(base_path, 'OUTPUT', program_name)
os.makedirs(output_path, exist_ok=True)

#SURVEY_PATH = os.path.join(base_path, 'DETECT_Data', 'HUF', 'kaye_365_huf_detect.csv')

SURVEY_PATH = os.path.join(base_path, 'DETECT_DATA_080825', 'HUF', 'DETECT_huf_2025-07-08_cleaned.csv')

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
log_step("Loading survey data.")
survey_df = pd.read_csv(SURVEY_PATH)

# Count before cleaning
num_subids_0 = survey_df['subid'].nunique()
total_rows_0 = len(survey_df)
log_step(f"Step 0: {num_subids_0} unique subids, {total_rows_0} total rows after merge.")
row_counts_0 = survey_df['subid'].value_counts().sort_index()
row_counts_0.to_csv(os.path.join(output_path, 'row_counts_step_0.csv'))
missing_0 = survey_df.isnull().sum()
missing_0[missing_0 > 0].to_csv(os.path.join(output_path, 'missing_counts_step_0.csv'))

#------ Step 1: Extract date and convert all dates to datetime.date------
log_step("Step 1: Converting all date columns to datetime and extracting 'date' column.")
# date_cols = [
#     'fall1_date', 'fall2_date', 'fall3_date',
#     'StartDate', 'hcru1_date', 'hcru2_date',
#     'acdt1_date', 'acdt2_date', 'acdt3_date',
#     'med1_date', 'med2_date', 'med3_date', 'med4_date'
# ]

date_cols = [
    'FALL1_DATE', 'FALL2_DATE', 'FALL3_DATE',
    'StartDate', 'HCRU1_DATE', 'HCRU2_DATE',
    'ACDT1_DATE', 'ACDT2_DATE', 'ACDT3_DATE',
    'MED1_DATE', 'MED2_DATE', 'MED3_DATE', 'MED4_DATE'
]

for col in date_cols:
    if col in survey_df.columns:
        survey_df[col] = pd.to_datetime(survey_df[col], errors='coerce').dt.date

survey_df['start_date'] = pd.to_datetime(survey_df['StartDate']).dt.date

num_subids_1 = survey_df['subid'].nunique()
total_rows_1 = len(survey_df)
log_step(f"Step 1: {num_subids_1} unique subids, {total_rows_1} total rows after date extraction and converting all date columns.")
row_counts_1 = survey_df['subid'].value_counts().sort_index()
row_counts_1.to_csv(os.path.join(output_path, 'row_counts_step_1.csv'))
missing_1 = survey_df.isnull().sum()
missing_1[missing_1 > 0].to_csv(os.path.join(output_path, 'missing_counts_step_1.csv'))

#------ Step 2: Drop missing values ------
log_step("Step 2: Dropping rows with missing subid or StartDate.")
survey_df = survey_df.dropna(subset=['subid', 'StartDate'])

num_subids_2 = survey_df['subid'].nunique()
total_rows_2 = len(survey_df)
log_step(f"Step 2: {num_subids_2} unique subids, {total_rows_2} total rows after dropping NAs.")
row_counts_2 = survey_df['subid'].value_counts().sort_index()
row_counts_2.to_csv(os.path.join(output_path, 'row_counts_step_2.csv'))
missing_2 = survey_df.isnull().sum()
missing_2[missing_2 > 0].to_csv(os.path.join(output_path, 'missing_counts_step_2.csv'))

#------ Step 3: Fill fall1_date if FALL == 1 ------
log_step("Step 3: Filling fall1_date from start_date if FALL == 1 and missing.")

num_falls = len(survey_df['FALL1_DATE'])

print(f"Number of falls recorded before dropping nulls: {num_falls}")

# Ensure date columns are datetime
survey_df['fall1_date'] = pd.to_datetime(survey_df['FALL1_DATE'], errors='coerce')
survey_df['fall2_date'] = pd.to_datetime(survey_df['FALL2_DATE'], errors='coerce')
survey_df['fall3_date'] = pd.to_datetime(survey_df['FALL3_DATE'], errors='coerce')
survey_df['start_date'] = pd.to_datetime(survey_df['start_date'], errors='coerce')

num_falls_1 = survey_df['fall1_date'].notna().sum().sum()

print(f"Number of falls recorded after dropping NAs: {num_falls_1}")

# # Fill missing fall1_date with (survey_date - 1 day) if fall == 1
# survey_df['fall1_date'] = survey_df.apply(
#     lambda row: row['fall1_date'] if pd.notnull(row['fall1_date']) 
#     else ((row['start_date'] - pd.Timedelta(days=1)) if row['FALL'] == 1 else pd.NaT),
#     axis=1
# )

#fill missing fall2_date and fall3_date
survey_df['fall2_date'] = survey_df.apply(
    lambda row: row['fall2_date'] if pd.notnull(row['fall2_date']) 
    else ((row['fall1_date'] + pd.Timedelta(days=1)) if row['Q5.6'] == 1 else pd.NaT),
    axis=1
)

survey_df['fall3_date'] = survey_df.apply(
    lambda row: row['fall3_date'] if pd.notnull(row['fall3_date']) 
    else ((row['fall2_date'] + pd.Timedelta(days=1)) if row['Q5.11'] == 1 else pd.NaT),
    axis=1
)

# # Ensure date columns are datetime
survey_df['fall1_date'] = pd.to_datetime(survey_df['fall1_date'], errors='coerce').dt.date
survey_df['fall2_date'] = pd.to_datetime(survey_df['fall2_date'], errors='coerce').dt.date
survey_df['fall3_date'] = pd.to_datetime(survey_df['fall3_date'], errors='coerce').dt.date
# survey_df['start_date'] = pd.to_datetime(survey_df['start_date'], errors='coerce').dt.date

#combine fall dates into a single column
survey_df['fall_dates'] = survey_df.apply(
    lambda row: sorted(
        [date for date in [row['fall1_date'], row['fall2_date'], row['fall3_date']] if pd.notnull(date)]
    ), axis=1
)


num_subids_3 = survey_df['subid'].nunique()
total_rows_3 = len(survey_df)
log_step(f"Step 3: {num_subids_3} unique subids, {total_rows_3} total rows after filling fall1_date.")
row_counts_3 = survey_df['subid'].value_counts().sort_index()
row_counts_3.to_csv(os.path.join(output_path, 'row_counts_step_3.csv'))
missing_3 = survey_df.isnull().sum()
missing_3[missing_3 > 0].to_csv(os.path.join(output_path, 'missing_counts_step_3.csv'))

#------ Step 4: Convert and Combine Hospital Dates ------
log_step("Step 4: Processing hospital visit dates.")

# Process  hospital event dates
survey_df['hospital_visit'] = pd.to_datetime(survey_df['HCRU1_DATE']).dt.date
survey_df['hospital_visit_2'] = pd.to_datetime(survey_df['HCRU2_DATE'], errors='coerce').dt.date

#combine hospital visit dates into a single column
survey_df['hospital_dates'] = survey_df.apply(
	lambda row: sorted(
		[date for date in [row['hospital_visit'], row['hospital_visit_2']] if pd.notnull(date)]
	), axis=1
)

num_subids_4 = survey_df['subid'].nunique()
total_rows_4 = len(survey_df)
log_step(f"Step 4: {num_subids_4} unique subids, {total_rows_4} total rows after aggregation.")
row_counts_4 = survey_df['subid'].value_counts().sort_index()
row_counts_4.to_csv(os.path.join(output_path, 'row_counts_step_4.csv'))
missing_4 = survey_df.isnull().sum()
missing_4[missing_4 > 0].to_csv(os.path.join(output_path, 'missing_counts_step_4.csv'))

# --------- Step 5: Combine Accident Dates ---------
log_step("Step 5: Processing accident dates.")

survey_df['ACDT1_DATE'] = pd.to_datetime(survey_df['ACDT1_DATE'], errors='coerce').dt.date
survey_df['ACDT2_DATE'] = pd.to_datetime(survey_df['ACDT2_DATE'], errors='coerce').dt.date
survey_df['ACDT3_DATE'] = pd.to_datetime(survey_df['ACDT3_DATE'], errors='coerce').dt.date

#combine accident dates into a single column
survey_df['accident_dates'] = survey_df.apply(
	lambda row: sorted(
		[date for date in [row['ACDT1_DATE'], row['ACDT2_DATE'], row['ACDT3_DATE']] if pd.notnull(date)]
	), axis=1
)

num_subids_5 = survey_df['subid'].nunique()
total_rows_5 = len(survey_df)
log_step(f"Step 5: {num_subids_5} unique subids, {total_rows_5} total rows after aggregation.")
row_counts_5 = survey_df['subid'].value_counts().sort_index()
row_counts_5.to_csv(os.path.join(output_path, 'row_counts_step_5.csv'))
missing_5 = survey_df.isnull().sum()
missing_5[missing_5 > 0].to_csv(os.path.join(output_path, 'missing_counts_step_5.csv'))

# --------- Step 6: Combine Medication Dates ---------
log_step("Step 6: Processing medication change dates.")

survey_df['MED1_DATE'] = pd.to_datetime(survey_df['MED1_DATE'], errors='coerce').dt.date
survey_df['MED2_DATE'] = pd.to_datetime(survey_df['MED2_DATE'], errors='coerce').dt.date
survey_df['MED3_DATE'] = pd.to_datetime(survey_df['MED3_DATE'], errors='coerce').dt.date
survey_df['MED4_DATE'] = pd.to_datetime(survey_df['MED4_DATE'], errors='coerce').dt.date

#combine accident dates into a single column
survey_df['medication_dates'] = survey_df.apply(
    lambda row: sorted(
        [date for date in [row['MED1_DATE'], row['MED2_DATE'], row['MED3_DATE'], row['MED4_DATE']] if pd.notnull(date)]
    ),
    axis=1
)

num_subids_6 = survey_df['subid'].nunique()
total_rows_6 = len(survey_df)
log_step(f"Step 6: {num_subids_6} unique subids, {total_rows_6} total rows after aggregation.")
row_counts_6 = survey_df['subid'].value_counts().sort_index()
row_counts_6.to_csv(os.path.join(output_path, 'row_counts_step_6.csv'))
missing_6 = survey_df.isnull().sum()
missing_6[missing_6 > 0].to_csv(os.path.join(output_path, 'missing_counts_step_6.csv'))


# --------- Step 7: Mood Event Dates ---------
log_step("Step 7: Processing mood-related event dates.")

#import mood blue data "Have you felt downhearted or blue for three or more days in the past week?" 1: yes 2: no
survey_df['mood_blue_date'] = survey_df.apply(
	lambda row: row['start_date'] - pd.Timedelta(days=1) if row['MOOD_BLUE'] == 1 else pd.NaT, axis=1
)

#import mood lonely "In the past week I felt lonely." 1: yes 2: no
survey_df['mood_lonely_date'] = survey_df.apply(
    lambda row: row['start_date'] - pd.Timedelta(days=1) if row['MOOD_LONV'] == 1 else pd.NaT, axis=1
)

num_subids_7 = survey_df['subid'].nunique()
total_rows_7 = len(survey_df)
log_step(f"Step 7: {num_subids_7} unique subids, {total_rows_7} total rows after aggregation.")
row_counts_7 = survey_df['subid'].value_counts().sort_index()
row_counts_7.to_csv(os.path.join(output_path, 'row_counts_step_7.csv'))
missing_7 = survey_df.isnull().sum()
missing_7[missing_7 > 0].to_csv(os.path.join(output_path, 'missing_counts_step_7.csv'))

# --------- Step 8: Daily survey metrics ---------
log_step("Step 8: Computing daily total survey time and submissions-per-day per subid.")
# 1) Coerce duration to numeric seconds (Qualtrics column: 'Duration (in seconds)')
duration_col = 'Duration (in seconds)'
if duration_col not in survey_df.columns:
    log_step(f"WARNING: '{duration_col}' column not found. Creating zeros.")
    survey_df[duration_col] = 0

survey_df['duration_sec_raw'] = pd.to_numeric(survey_df[duration_col], errors='coerce').fillna(0)

# 2) Build per-day aggregation key (subid, start_date)
metrics = (
    survey_df
    .assign(start_date=survey_df['start_date'])  # ensure present
    .groupby(['subid', 'start_date'], dropna=False)
    .agg(
        daily_duration_seconds=('duration_sec_raw', 'sum'),
        submissions_per_day=('duration_sec_raw', 'size')
    )
    .reset_index()
)

# 3) Save tidy per-day metrics table
metrics_out = os.path.join(output_path, 'survey_daily_metrics.csv')
metrics.to_csv(metrics_out, index=False)
log_step(f"Step 8: Saved per-day metrics to {metrics_out}")

# 4) Merge back onto each row for easy modeling/QA
survey_df = survey_df.merge(
    metrics,
    on=['subid', 'start_date'],
    how='left'
)

num_subids_8 = survey_df['subid'].nunique()
total_rows_8 = len(survey_df)
log_step(f"Step 8: {num_subids_8} unique subids, {total_rows_8} total rows after merging metrics.")
row_counts_8 = survey_df['subid'].value_counts().sort_index()
row_counts_8.to_csv(os.path.join(output_path, 'row_counts_step_8.csv'))
missing_8 = survey_df.isnull().sum()
missing_8[missing_8 > 0].to_csv(os.path.join(output_path, 'missing_counts_step_8.csv'))

# --------- Step 9: Minimal status mapping ---------
log_step("Step 9: Mapping survey event flags to status codes (1=yes, 2=no, else=unknown).")

# Define unknown code
UNKNOWN_CODE = 99

# Only create status columns for flags that actually exist in the file
if 'FALL' in survey_df.columns:
    survey_df['FALL_status_code'] = survey_df['FALL'].map({1:1, 2:2}).fillna(UNKNOWN_CODE).astype(int)
    survey_df['FALL_status']      = survey_df['FALL_status_code'].map({1:'yes', 2:'no', UNKNOWN_CODE:'unknown'})

if 'HCRU' in survey_df.columns:
    survey_df['HCRU_status_code'] = survey_df['HCRU'].map({1:1, 2:2}).fillna(UNKNOWN_CODE).astype(int)
    survey_df['HCRU_status']      = survey_df['HCRU_status_code'].map({1:'yes', 2:'no', UNKNOWN_CODE:'unknown'})

if 'ACDT' in survey_df.columns:
    survey_df['ACDT_status_code'] = survey_df['ACDT'].map({1:1, 2:2}).fillna(UNKNOWN_CODE).astype(int)
    survey_df['ACDT_status']      = survey_df['ACDT_status_code'].map({1:'yes', 2:'no', UNKNOWN_CODE:'unknown'})

if 'MED' in survey_df.columns:
    survey_df['MED_status_code'] = survey_df['MED'].map({1:1, 2:2}).fillna(UNKNOWN_CODE).astype(int)
    survey_df['MED_status']      = survey_df['MED_status_code'].map({1:'yes', 2:'no', UNKNOWN_CODE:'unknown'})

if 'MOOD_BLUE' in survey_df.columns:
    survey_df['MOOD_BLUE_status_code'] = survey_df['MOOD_BLUE'].map({1:1, 2:2}).fillna(UNKNOWN_CODE).astype(int)
    survey_df['MOOD_BLUE_status']      = survey_df['MOOD_BLUE_status_code'].map({1:'yes', 2:'no', UNKNOWN_CODE:'unknown'})

if 'MOOD_LONV' in survey_df.columns:
    survey_df['MOOD_LONV_status_code'] = survey_df['MOOD_LONV'].map({1:1, 2:2}).fillna(UNKNOWN_CODE).astype(int)
    survey_df['MOOD_LONV_status']      = survey_df['MOOD_LONV_status_code'].map({1:'yes', 2:'no', UNKNOWN_CODE:'unknown'})

# Summary logging for this step
num_subids_9 = survey_df['subid'].nunique()
total_rows_9 = len(survey_df)
log_step(f"Step 9: {num_subids_9} unique subids, {total_rows_9} total rows after status mapping.")

row_counts_9 = survey_df['subid'].value_counts().sort_index()
row_counts_9.to_csv(os.path.join(output_path, 'row_counts_step_9.csv'))

missing_9 = survey_df.isnull().sum()
missing_9[missing_9 > 0].to_csv(os.path.join(output_path, 'missing_counts_step_9.csv'))

# --------- Step 9.5: Daily status counts (yes/no/unknown) per event ----------
log_step("Step 9.5: Building per-day status counts (yes/no/unknown) for each event.")

# We'll produce one tidy table per (subid, start_date) with columns like:
#   FALL_yes, FALL_no, FALL_unknown, HCRU_yes, ..., MOOD_LONV_unknown
# If an event flag wasn't present that day, counts are 0.

# 0) Base (all unique subid × day pairs)
daily_status = survey_df[['subid', 'start_date']].drop_duplicates().copy()

# 1) Figure out which events we actually have status for
event_status_cols = []
if 'FALL_status_code' in survey_df.columns:       event_status_cols.append(('FALL', 'FALL_status_code'))
if 'HCRU_status_code' in survey_df.columns:       event_status_cols.append(('HCRU', 'HCRU_status_code'))
if 'ACDT_status_code' in survey_df.columns:       event_status_cols.append(('ACDT', 'ACDT_status_code'))
if 'MED_status_code' in survey_df.columns:        event_status_cols.append(('MED', 'MED_status_code'))
if 'MOOD_BLUE_status_code' in survey_df.columns:  event_status_cols.append(('MOOD_BLUE', 'MOOD_BLUE_status_code'))
if 'MOOD_LONV_status_code' in survey_df.columns:  event_status_cols.append(('MOOD_LONV', 'MOOD_LONV_status_code'))

# 2) For each event, count how many 1/2/99 responses occurred that day
for ev, code_col in event_status_cols:
    tmp = survey_df[['subid', 'start_date', code_col]].dropna(subset=['subid', 'start_date'])
    tmp = tmp.rename(columns={code_col: 'code'})

    # count of each code per subid × day
    counts = (
        tmp.groupby(['subid', 'start_date', 'code'])
           .size()
           .unstack(fill_value=0)
           .reset_index()
    )

    # ensure all three columns exist (1=yes, 2=no, 99=unknown)
    for needed in [1, 2, 99]:
        if needed not in counts.columns:
            counts[needed] = 0

    # rename to readable column names
    counts = counts.rename(columns={
        1:   f'{ev}_yes',
        2:   f'{ev}_no',
        99:  f'{ev}_unknown'
    })

    # keep only the columns we need
    counts = counts[['subid', 'start_date', f'{ev}_yes', f'{ev}_no', f'{ev}_unknown']]

    # merge into the daily_status table
    daily_status = daily_status.merge(counts, on=['subid', 'start_date'], how='left')

# 3) Fill any missing counts with 0
count_cols = [c for c in daily_status.columns if c.endswith('_yes') or c.endswith('_no') or c.endswith('_unknown')]
daily_status[count_cols] = daily_status[count_cols].fillna(0).astype(int)

# 4) Save per-day status table
daily_status_out = os.path.join(output_path, 'survey_daily_status.csv')
daily_status.sort_values(['subid', 'start_date']).to_csv(daily_status_out, index=False)
log_step(f"Step 9.5: Saved per-day status counts to {daily_status_out}")

#------ Step 10: Derive Composite Fall Labels (injury, hospital, terminal) ------
log_step("Step 10: Creating composite fall labels (injury, hospital, terminal).")

#ensure FALL, HCRU and FALL1_INJ are numeric (1=yes, 2=no)
for c in ['FALL_status_code','HCRU_status_code','FALL1_INJ']:
	if c in survey_df.columns:
		survey_df[c] = pd.to_numeric(survey_df[c], errors='coerce')

#---1. FAll with Injury---
survey_df['fall_with_injury'] = np.where(
	(survey_df.get('FALL1_INJ', 0) == 1) & (survey_df.get('FALL_status_code', 0) == 1),
    1,0
)

#---2. Fall with Hospital Visit---
survey_df['fall_with_hospital'] = np.where(
	(survey_df.get('FALL_status_code', 0) == 1) & (survey_df.get('HCRU_status_code', 0) == 1),
	1,0
)

#---3. Fall with Terminal Event---
survey_df['fall_terminal'] = np.where(
    (survey_df['fall_with_injury'] == 1) | (survey_df['fall_with_hospital'] == 1),
    1, 0
)

# Log counts after consolidation
num_subids_10 = survey_df['subid'].nunique()
total_rows_10 = len(survey_df)
log_step(f"Step 10: {num_subids_10} unique subids, {total_rows_10} total rows after consolidation to one row per day.")
row_counts_10 = survey_df['subid'].value_counts().sort_index()
row_counts_10.to_csv(os.path.join(output_path, 'row_counts_step_10.csv'))
missing_10 = survey_df.isnull().sum()
missing_10[missing_10 > 0].to_csv(os.path.join(output_path, 'missing_counts_step_10.csv'))

#------4. Logging Summary ----
n_injury = survey_df['fall_with_injury'].sum()
n_hospital = survey_df['fall_with_hospital'].sum()
n_terminal = survey_df['fall_terminal'].sum()
log_step(f"Step 12 summary: {n_injury} fall_with_injury events, {n_hospital} fall_with_hospital events, {n_terminal} fall_terminal events.")

#------ Save final before dropping duplicates ------
output_final = os.path.join(output_path, 'survey_cleaned.csv')
survey_df.to_csv(output_final, index=False)
log_step(f"Cleaned before dropping duplicates survey data saved to: {output_final}")

#------ Step 11: Check for duplicates ------
log_step("Step 11: Checking for duplicate (subid, date) rows.")
duplicates = survey_df.duplicated(subset=['subid', 'start_date'], keep=False)
dup_rows = survey_df[duplicates]
if not dup_rows.empty:
    log_step(f"Found {len(dup_rows)} duplicate rows. Saving to 'duplicate_entries.csv'.")
    dup_rows.to_csv(os.path.join(output_path, 'duplicate_entries.csv'), index=False)
else:
    log_step("No duplicate rows found.")

num_falls_1 = survey_df['fall1_date'].notna().sum().sum()

print(f"Number of falls recorded in step 10: {num_falls_1}")

#------ Step 11: Drop duplicates  ------
survey_df_copy = survey_df.drop_duplicates(subset=['subid', 'start_date'], keep='first').copy()

log_step("Checking for duplicate (subid, date) rows.")
duplicates = survey_df_copy.duplicated(subset=['subid', 'start_date'], keep=False)
dup_rows = survey_df_copy[duplicates]
if not dup_rows.empty:
    log_step(f"Found {len(dup_rows)} duplicate rows after dropping duplicates. Saving to 'duplicate_entries_after_dropping.csv'.")
    dup_rows.to_csv(os.path.join(output_path, 'duplicate_entries_after_dropping.csv'), index=False)
else:
    log_step("No duplicate rows found.")
    
# Log counts after consolidation
num_subids_11 = survey_df_copy['subid'].nunique()
total_rows_11 = len(survey_df_copy)
log_step(f"Step 11: {num_subids_11} unique subids, {total_rows_11} total rows after dropping duplicates to one row per day.")
row_counts_11 = survey_df_copy['subid'].value_counts().sort_index()
row_counts_11.to_csv(os.path.join(output_path, 'row_counts_step_11.csv'))
missing_11 = survey_df_copy.isnull().sum()
missing_11[missing_11 > 0].to_csv(os.path.join(output_path, 'missing_counts_step_11.csv'))

#------ Save final unconsolidated cleaned data ------
output_final = os.path.join(output_path, 'survey_cleaned_unconsolidated.csv')
survey_df_copy.to_csv(output_final, index=False)
log_step(f"Cleaned unconsolidated survey data saved to: {output_final}")

#------ Step 12: Consolidate to one row per (subid, start_date) ------
log_step("Step 12: Consolidating multiple same-day submissions into a single record per (subid, start_date) [aggregate].")

import numpy as np

# 0) Ensure types are consistent BEFORE any groupby aggregation
#    a) Scalar date columns -> Timestamp
scalar_date_cols = ['mood_blue_date','mood_lonely_date','fall1_date','fall2_date','fall3_date', 'MED1_DATE', 'MED2_DATE', 'MED3_DATE', 'MED4_DATE',
                    'hospital_visit', 'hospital_visit_2', 'ACDT1_DATE', 'ACDT2_DATE', 'ACDT3_DATE']
for c in scalar_date_cols:
    if c in survey_df.columns:
        survey_df[c] = pd.to_datetime(survey_df[c], errors='coerce')

#    b) StartDate -> Timestamp (for representative earliest-of-day)
if 'StartDate' in survey_df.columns:
    survey_df['StartDate'] = pd.to_datetime(survey_df['StartDate'], errors='coerce')

#    c) Event flags to numeric (handle "1"/"2" strings)
for c in ['FALL','HCRU','ACDT','MED','MOOD_BLUE','MOOD_LONV', 'FALL1_INJ', 'fall_with_injury', 'fall_with_hospital', 'fall_terminal']:
    if c in survey_df.columns:
        survey_df[c] = pd.to_numeric(survey_df[c], errors='coerce')

# 1) Group
grp = survey_df.groupby(['subid','start_date'], dropna=False)

# 2) Build base (submissions_per_day)
survey_daily_df = grp.size().rename('submissions_per_day').reset_index()

# 3) Attach daily duration/counts already computed in Step 8
if 'daily_duration_seconds' in metrics.columns:
    survey_daily_df = survey_daily_df.merge(
        metrics[['subid','start_date','daily_duration_seconds','submissions_per_day']],
        on=['subid','start_date'],
        how='left',
        suffixes=('', '_m')
    )
    # prefer the metrics values
    if 'submissions_per_day_m' in survey_daily_df.columns:
        survey_daily_df['submissions_per_day'] = survey_daily_df['submissions_per_day_m'].fillna(survey_daily_df['submissions_per_day'])
        survey_daily_df = survey_daily_df.drop(columns=[c for c in survey_daily_df.columns if c.endswith('_m')])

# 4) Collapse flags: any yes(1) > any no(2) > else 99
for flag in ['FALL','HCRU','ACDT','MED','MOOD_BLUE','MOOD_LONV', 'FALL1_INJ', 'fall_with_injury', 'fall_with_hospital', 'fall_terminal']:
    if flag in survey_df.columns:
        collapsed = grp[flag].apply(lambda s: 1 if (s == 1).any()
                                             else (2 if (s == 2).any() else 99))
        survey_daily_df = survey_daily_df.merge(
            collapsed.rename(flag).reset_index(),
            on=['subid','start_date'], how='left'
        )

# 5) Union + sort list-of-dates, coercing each element to Timestamp
for lstcol in ['fall_dates','hospital_dates','accident_dates','medication_dates']:
    if lstcol in survey_df.columns:
        merged_lists = grp[lstcol].apply(
            lambda s: sorted({
                pd.to_datetime(x, errors='coerce') 
                for v in s.dropna()
                for x in (v if isinstance(v, list) else [v])
                if pd.notna(x)
            })
        )
        survey_daily_df = survey_daily_df.merge(
            merged_lists.rename(lstcol).reset_index(),
            on=['subid','start_date'], how='left'
        )

# 6) Scalar dates: take earliest Timestamp that day
for dcol in scalar_date_cols:
    if dcol in survey_df.columns:
        earliest = grp[dcol].min()  # now safe because all Timestamps or NaT
        survey_daily_df = survey_daily_df.merge(
            earliest.rename(dcol).reset_index(),
            on=['subid','start_date'], how='left'
        )

# 7) Representative StartDate: earliest timestamp that day (if present)
if 'StartDate' in survey_df.columns:
    rep_start = grp['StartDate'].min()
    survey_daily_df = survey_daily_df.merge(
        rep_start.rename('StartDate').reset_index(),
        on=['subid','start_date'], how='left'
    )

# 8) Stable demographics: first non-null
for stable in ['sex','age','age_at_visit','amyloid','hispanic','marital','alzdis_Present']:
    if stable in survey_df.columns:
        first_nn = grp[stable].apply(lambda s: next((x for x in s if pd.notna(x)), pd.NA))
        survey_daily_df = survey_daily_df.merge(
            first_nn.rename(stable).reset_index(),
            on=['subid','start_date'], how='left'
        )

# 9) Replace working df with consolidated version
survey_df = survey_daily_df.sort_values(['subid','start_date']).reset_index(drop=True)

# Log counts after consolidation
num_subids_12 = survey_df['subid'].nunique()
total_rows_12 = len(survey_df)
log_step(f"Step 12: {num_subids_12} unique subids, {total_rows_12} total rows after consolidation to one row per day.")
row_counts_12 = survey_df['subid'].value_counts().sort_index()
row_counts_12.to_csv(os.path.join(output_path, 'row_counts_step_12.csv'))
missing_12 = survey_df.isnull().sum()
missing_12[missing_12 > 0].to_csv(os.path.join(output_path, 'missing_counts_step_12.csv'))

""" #------ Step 12: Derive Composite Fall Labels (injury, hospital, terminal) ------
log_step("Creating composite fall labels (injury, hospital, terminal).")

#ensure FALL, HCRU and FALL1_INJ are numeric (1=yes, 2=no)
for c in ['FALL','HCRU','FALL1_INJ']:
	if c in survey_df.columns:
		survey_df[c] = pd.to_numeric(survey_df[c], errors='coerce')

#---1. FAll with Injury---
survey_df['fall_with_injury'] = np.where(
	(survey_df.get('FALL1_INJ', 0) == 1) & (survey_df.get('FALL', 0) == 1),
    1,0
)

#---2. Fall with Hospital Visit---
survey_df['fall_with_hospital'] = np.where(
	(survey_df.get('FALL', 0) == 1) & (survey_df.get('HCRU', 0) == 1),
	1,0
)

#---3. Fall with Terminal Event---
survey_df['fall_terminal'] = np.where(
    (survey_df['fall_with_injury'] == 1) | (survey_df['fall_with_hospital'] == 1),
    1, 0
)

# Log counts after consolidation
num_subids_12 = survey_df['subid'].nunique()
total_rows_12 = len(survey_df)
log_step(f"Step 12: {num_subids_12} unique subids, {total_rows_12} total rows after consolidation to one row per day.")
row_counts_12 = survey_df['subid'].value_counts().sort_index()
row_counts_12.to_csv(os.path.join(output_path, 'row_counts_step_12.csv'))
missing_12 = survey_df.isnull().sum()
missing_12[missing_12 > 0].to_csv(os.path.join(output_path, 'missing_counts_step_12.csv'))

#------4. Logging Summary ----
n_injury = survey_df['fall_with_injury'].sum()
n_hospital = survey_df['fall_with_hospital'].sum()
n_terminal = survey_df['fall_terminal'].sum()
log_step(f"Step 12 summary: {n_injury} fall_with_injury events, {n_hospital} fall_with_hospital events, {n_terminal} fall_terminal events.")
 """
#------ Step 13: Check for duplicates after consolidation ------
log_step("Checking for duplicate (subid, date) rows after consolidation.")
duplicates = survey_df.duplicated(subset=['subid', 'start_date'], keep=False)
dup_rows = survey_df[duplicates]
if not dup_rows.empty:
    log_step(f"Found {len(dup_rows)} duplicate rows. Saving to 'duplicate_entries_after_consolidation.csv'.")
    dup_rows.to_csv(os.path.join(output_path, 'duplicate_entries_after_consolidation.csv'), index=False)
else:
    log_step("No duplicate rows found.")

# Save the consolidated daily dataset (be explicit)
output_daily = os.path.join(output_path, 'survey_cleaned_consolidated.csv')
survey_df.to_csv(output_daily, index=False)
log_step(f"Step 13: Saved consolidated daily dataset to: {output_daily}")
 
#------ Final Summary Table ------
summary = pd.DataFrame({
                'unique_subids': [num_subids_0, num_subids_1, num_subids_2, num_subids_3, num_subids_4, num_subids_5, num_subids_6, num_subids_7, num_subids_8, num_subids_9, num_subids_10, num_subids_11, num_subids_12],
                'n_rows':        [total_rows_0, total_rows_1, total_rows_2, total_rows_3, total_rows_4, total_rows_5, total_rows_6, total_rows_7, total_rows_8, total_rows_9, total_rows_10, total_rows_11, total_rows_12]
}, index=['step_0_initial', 'step_1_extracted_date', 'step_2_drop_na', 'step_3_fill_fall_dates', 'step_4_hospital_dates', 'step_5_accident_dates', 'step_6_medication_dates', 'step_7_mood_dates', 'step_8_daily_metrics', 'step_9_per_day_status', 'step_10_composite_fall_labels','step_11_drop_duplicates',  'step_12_consolidation', ])

summary.to_csv(os.path.join(output_path, 'summary_pipeline_overview.csv'))
print("Summary saved to summary_pipeline_overview.csv")

#------ Final Missingness Summary ------
missing_all = pd.DataFrame({
    'step_0': missing_0,
    'step_1': missing_1,
    'step_2': missing_2,
    'step_3': missing_3,
    'step_4': missing_4,
    'step_6': missing_6,
    'step_7': missing_7,
    'step_8': missing_8,
    'step_9': missing_9,
    'step_10': missing_10,
    'step_11': missing_11,
    'step_12': missing_12
}).fillna(0).astype(int)

missing_all.to_csv(os.path.join(output_path, 'missing_counts_summary.csv'))
print("Missing value summary saved to missing_counts_summary.csv")

#------ Step 14: Generate Weekly Frequency Tables ------
log_step("Step 14: Generating weekly frequency tables for all events.")

def explode_event(df, col, label):
    exploded = df[['subid', col]].explode(col).dropna()
    exploded[col] = pd.to_datetime(exploded[col], errors='coerce')
    exploded['week'] = exploded[col].dt.to_period('W').apply(lambda r: r.start_time)
    freq = exploded.groupby('week').size().reset_index(name=f'{label}_count')
    return freq

# List-based events
fall_freq     = explode_event(survey_df, 'fall_dates',       'fall')
hospital_freq = explode_event(survey_df, 'hospital_dates',   'hospital')
accident_freq = explode_event(survey_df, 'accident_dates',   'accident')
med_freq      = explode_event(survey_df, 'medication_dates', 'medication')
fall_injury_freq = explode_event(
	survey_df[survey_df['fall_with_injury'] == 1], 'fall_dates', 'fall_injury'
)
fall_hospital_freq = explode_event(
	survey_df[survey_df['fall_with_hospital'] == 1], 'fall_dates', 'fall_hospital'
)
fall_terminal_freq = explode_event(
	survey_df[survey_df['fall_terminal'] == 1], 'fall_dates', 'fall_terminal'
)

# Scalar mood events (treat as single dates)
def scalar_event(df, date_col, label):
    tmp = df[['subid', date_col]].dropna()
    tmp[date_col] = pd.to_datetime(tmp[date_col])
    tmp['week'] = tmp[date_col].dt.to_period('W').apply(lambda r: r.start_time)
    freq = tmp.groupby('week').size().reset_index(name=f'{label}_count')
    return freq

mood_blue_freq   = scalar_event(survey_df, 'mood_blue_date', 'blue_mood')
mood_lonely_freq = scalar_event(survey_df, 'mood_lonely_date', 'lonely_mood')

#------ Save frequency tables ------
freq_output_path = os.path.join(output_path, 'event_frequency')
os.makedirs(freq_output_path, exist_ok=True)

fall_freq.to_csv(os.path.join(freq_output_path, 'fall_weekly.csv'), index=False)
hospital_freq.to_csv(os.path.join(freq_output_path, 'hospital_weekly.csv'), index=False)
accident_freq.to_csv(os.path.join(freq_output_path, 'accident_weekly.csv'), index=False)
med_freq.to_csv(os.path.join(freq_output_path, 'medication_weekly.csv'), index=False)
mood_blue_freq.to_csv(os.path.join(freq_output_path, 'blue_mood_weekly.csv'), index=False)
mood_lonely_freq.to_csv(os.path.join(freq_output_path, 'lonely_mood_weekly.csv'), index=False)
fall_injury_freq.to_csv(os.path.join(freq_output_path, 'fall_injury_weekly.csv'), index=False)
fall_hospital_freq.to_csv(os.path.join(freq_output_path, 'fall_hospital_weekly.csv'), index=False)
fall_terminal_freq.to_csv(os.path.join(freq_output_path, 'fall_terminal_weekly.csv'), index=False)

log_step("Weekly frequency tables saved to event_frequency/ directory.")

def plot_frequency(df, count_col, event_name):
    plt.figure(figsize=(12, 4))
    sns.barplot(data=df, x='week', y=count_col, color='steelblue')
    plt.xticks(rotation=45, ha='right')
    plt.title(f'{event_name.capitalize()} Events Per Week')
    plt.xlabel('Week Starting')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(os.path.join(freq_output_path, f'{event_name}_weekly_plot.png'))
    plt.close()

plot_frequency(fall_freq, 'fall_count', 'fall')
plot_frequency(hospital_freq, 'hospital_count', 'hospital')
plot_frequency(accident_freq, 'accident_count', 'accident')
plot_frequency(med_freq, 'medication_count', 'medication')
plot_frequency(mood_blue_freq, 'blue_mood_count', 'blue_mood')
plot_frequency(mood_lonely_freq, 'lonely_mood_count', 'lonely_mood')
plot_frequency(fall_injury_freq, 'fall_injury_count', 'fall_injury')
plot_frequency(fall_hospital_freq, 'fall_hospital_count', 'fall_hospital')
plot_frequency(fall_terminal_freq, 'fall_terminal_count', 'fall_terminal')

log_step("Weekly plots saved as PNGs in event_frequency/.")
