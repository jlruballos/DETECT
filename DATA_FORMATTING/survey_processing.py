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

program_name = 'survey_processing'

#------ Set up paths ------
base_path = '/mnt/d/DETECT'
output_path = os.path.join(base_path, 'OUTPUT', program_name)
os.makedirs(output_path, exist_ok=True)

SURVEY_PATH = os.path.join(base_path, 'DETECT_Data', 'HUF', 'kaye_365_huf_detect.csv')

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
date_cols = [
    'fall1_date', 'fall2_date', 'fall3_date',
    'StartDate', 'hcru1_date', 'hcru2_date',
    'acdt1_date', 'acdt2_date', 'acdt3_date',
    'med1_date', 'med2_date', 'med3_date', 'med4_date'
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
# Ensure date columns are datetime
survey_df['fall1_date'] = pd.to_datetime(survey_df['fall1_date'], errors='coerce')
survey_df['fall2_date'] = pd.to_datetime(survey_df['fall2_date'], errors='coerce')
survey_df['fall3_date'] = pd.to_datetime(survey_df['fall3_date'], errors='coerce')
survey_df['start_date'] = pd.to_datetime(survey_df['StartDate'], errors='coerce')

# Fill missing fall1_date with (survey_date - 1 day) if fall == 1
survey_df['fall1_date'] = survey_df.apply(
    lambda row: row['fall1_date'].date() if pd.notnull(row['fall1_date']) 
    else ((row['start_date'] - pd.Timedelta(days=1)).date() if row['FALL'] == 1 else pd.NaT),
    axis=1
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
survey_df['hospital_visit'] = pd.to_datetime(survey_df['hcru1_date']).dt.date
survey_df['hospital_visit_2'] = pd.to_datetime(survey_df['hcru2_date'], errors='coerce').dt.date

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

survey_df['ACDT1_DATE'] = pd.to_datetime(survey_df['acdt1_date'], errors='coerce').dt.date
survey_df['ACDT2_DATE'] = pd.to_datetime(survey_df['acdt2_date'], errors='coerce').dt.date
survey_df['ACDT3_DATE'] = pd.to_datetime(survey_df['acdt3_date'], errors='coerce').dt.date

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

survey_df['MED1_DATE'] = pd.to_datetime(survey_df['med1_date'], errors='coerce').dt.date
survey_df['MED2_DATE'] = pd.to_datetime(survey_df['med2_date'], errors='coerce').dt.date
survey_df['MED3_DATE'] = pd.to_datetime(survey_df['med3_date'], errors='coerce').dt.date
survey_df['MED4_DATE'] = pd.to_datetime(survey_df['med4_date'], errors='coerce').dt.date

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

#------ Step 8: Check for duplicates ------
log_step("Checking for duplicate (subid, date) rows.")
duplicates = survey_df.duplicated(subset=['subid', 'start_date'], keep=False)
dup_rows = survey_df[duplicates]
if not dup_rows.empty:
    log_step(f"Found {len(dup_rows)} duplicate rows. Saving to 'duplicate_entries.csv'.")
    dup_rows.to_csv(os.path.join(output_path, 'duplicate_entries.csv'), index=False)
else:
    log_step("No duplicate rows found.")
    

#------ Save final cleaned data ------
output_final = os.path.join(output_path, 'survey_cleaned.csv')
survey_df.to_csv(output_final, index=False)
log_step(f"Cleaned survey data saved to: {output_final}")

#------ Final Summary Table ------
summary = pd.DataFrame({
                'unique_subids': [num_subids_0, num_subids_1, num_subids_2, num_subids_3, num_subids_4, num_subids_5, num_subids_6, num_subids_7],
                'n_rows':        [total_rows_0, total_rows_1, total_rows_2, total_rows_3, total_rows_4, total_rows_5, total_rows_6, total_rows_7]
}, index=['step_0_initial', 'step_1_extracted_date', 'step_2_drop_na', 'step_3_fill_fall_dates', 'step_4_hospital_dates', 'step_5_accident_dates', 'step_6_medication_dates', 'step_7_mood_dates'])

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
    'step_7': missing_7
}).fillna(0).astype(int)

missing_all.to_csv(os.path.join(output_path, 'missing_counts_summary.csv'))
print("Missing value summary saved to missing_counts_summary.csv")

#------ Step 9: Generate Weekly Frequency Tables ------
log_step("Step 9: Generating weekly frequency tables for all events.")

def explode_event(df, col, label):
    exploded = df[['subid', col]].explode(col).dropna()
    exploded[col] = pd.to_datetime(exploded[col], errors='coerce')
    exploded['week'] = exploded[col].dt.to_period('W').apply(lambda r: r.start_time)
    freq = exploded.groupby('week').size().reset_index(name=f'{label}_count')
    return freq

# List-based events
fall_freq     = explode_event(survey_df, 'fall1_date',       'fall')
hospital_freq = explode_event(survey_df, 'hospital_dates',   'hospital')
accident_freq = explode_event(survey_df, 'accident_dates',   'accident')
med_freq      = explode_event(survey_df, 'medication_dates', 'medication')

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

log_step("Weekly plots saved as PNGs in event_frequency/.")
