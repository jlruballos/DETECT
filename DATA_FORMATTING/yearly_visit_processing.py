#!/usr/bin/env python3
"""
Yearly Clinical Visit Processing Pipeline

This script processes annual clinical visit data from the DETECT study cohort. It performs 
data cleaning, demographic harmonization, feature engineering, and categorical recoding 
to prepare the dataset for downstream modeling and descriptive analyses.

Main steps include:
- Loading raw clinical visit data
- Renaming and formatting core identifiers (e.g., subid, visit year)
- Generating visit-level date columns
- Filling missing Alzheimer diagnosis fields
- Forward-filling stable demographic features across visits per participant
- Creating mean MoCA score per participant
- Recoding:
    - Age into 5-year buckets (e.g., 55-59, 60-64)
    - Education into categories: Below HS, HS, Some College, College+
    - MoCA scores into cognitive bands: Normal, MCI, Moderate, Severe
    - Race into 3-level group: White, Non-White, Unknown
- Checking for and saving duplicate entries
- Saving cleaned and recoded datasets

Outputs include:
- Cleaned dataset (`yearly_cleaned.csv`)
- Recoded dataset (`yearly_recoded.csv`)
- Categorical value counts (e.g., age buckets, education, MoCA category, race group)
- Summary tables of row counts and missing values at each processing step
- Logged pipeline steps with timestamps
"""

__author__ = "Jorge Ruballos"
__email__ = "ruballoj@oregonstate.edu"
__date__ = "2025-07-28"
__version__ = "1.3.0"

import pandas as pd
import os
from datetime import datetime

program_name = 'yearly_visit_processing'

#------ Set up paths ------
base_path = '/mnt/d/DETECT'
output_path = os.path.join(base_path, 'OUTPUT', program_name)
os.makedirs(output_path, exist_ok=True)

YEARLY_PATH = os.path.join(base_path, 'DETECT_Data', 'Clinical', 'Clinical', 'kaye_365_clin_age_at_visit.csv')

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
yearly_df = pd.read_csv(YEARLY_PATH)

#--------RENAME COLUMNS --------
yearly_df = yearly_df.rename(columns={'record_id': 'subid'})

# Count before cleaning
num_subids_0 = yearly_df['subid'].nunique()
total_rows_0 = len(yearly_df)
log_step(f"Step 0: {num_subids_0} unique subids, {total_rows_0} total rows after merge.")
row_counts_0 = yearly_df['subid'].value_counts().sort_index()
row_counts_0.to_csv(os.path.join(output_path, 'row_counts_step_0.csv'))
missing_0 = yearly_df.isnull().sum()
missing_0[missing_0 > 0].to_csv(os.path.join(output_path, 'missing_counts_step_0.csv'))

#------ Step 1: Drop missing values ------
log_step("Step 1: Dropping rows with missing subid or Visit Year.")
yearly_df = yearly_df.dropna(subset=['subid', 'visityr_cr'])

num_subids_1 = yearly_df['subid'].nunique()
total_rows_1 = len(yearly_df)
log_step(f"Step 1: {num_subids_1} unique subids, {total_rows_1} total rows after dropping NAs.")
row_counts_1 = yearly_df['subid'].value_counts().sort_index()
row_counts_1.to_csv(os.path.join(output_path, 'row_counts_step_1.csv'))
missing_1 = yearly_df.isnull().sum()
missing_1[missing_1 > 0].to_csv(os.path.join(output_path, 'missing_counts_step_1.csv'))

#------ Step 2: Create a date for the first day of the visit year------
log_step("Step 2: Create a date for the first day of the visit year")

# Create a date for the first day of the visit year
yearly_df = yearly_df[yearly_df['visityr_cr'].notna()]
yearly_df['visit_date'] = pd.to_datetime(yearly_df['visityr_cr'].astype(int).astype(str) + '-01-01')

print(f"Number of subjects with yearly survey data: {len(yearly_df['subid'].unique())}")
num_subids_2 = yearly_df['subid'].nunique()
total_rows_2 = len(yearly_df)
log_step(f"Step 2: {num_subids_2} unique subids, {total_rows_2} total rows after date extraction and converting all date columns.")
row_counts_2 = yearly_df['subid'].value_counts().sort_index()
row_counts_2.to_csv(os.path.join(output_path, 'row_counts_step_2.csv'))
missing_2 = yearly_df.isnull().sum()
missing_2[missing_2 > 0].to_csv(os.path.join(output_path, 'missing_counts_step_2.csv'))

#------- Step 3: for alzdis fill in missing with 0------
log_step("Step 3: Filling missing alzdis with 0.")
yearly_df['alzdis'] = yearly_df['alzdis'].fillna(0)

print(f"Number of subjects with yearly survey data: {len(yearly_df['subid'].unique())}")
num_subids_3 = yearly_df['subid'].nunique()
total_rows_3 = len(yearly_df)
log_step(f"Step 3: {num_subids_3} unique subids, {total_rows_3} total rows after filling missing alzdis.")
row_counts_3 = yearly_df['subid'].value_counts().sort_index()
row_counts_3.to_csv(os.path.join(output_path, 'row_counts_step_3.csv'))
missing_3 = yearly_df.isnull().sum()
missing_3[missing_3 > 0].to_csv(os.path.join(output_path, 'missing_counts_step_3.csv'))

#------ Step 4: Create MOCA average column ------
log_step("Step 4: Creating MOCA average column.")
#log the number of unique subids in demo_df after filtering for visit year

# Process MOCA scores: average per subject
yearly_df['moca_avg'] = yearly_df.groupby('subid')['mocatots'].transform('mean')

print(f"Number of subjects with yearly survey data: {len(yearly_df['subid'].unique())}")
num_subids_4 = yearly_df['subid'].nunique()
total_rows_4 = len(yearly_df)
log_step(f"Step 4: {num_subids_4} unique subids, {total_rows_4} total rows after MOCA average calculation.")
row_counts_4 = yearly_df['subid'].value_counts().sort_index()
row_counts_4.to_csv(os.path.join(output_path, 'row_counts_step_4.csv'))
missing_4 = yearly_df.isnull().sum()
missing_4[missing_4 > 0].to_csv(os.path.join(output_path, 'missing_counts_step_4.csv'))

#------ Step 5: Forward fill certain demographics that wont change over time ------
log_step("Step 5: Forward filling demographic features per participant.")
demographic_features = ['hispanic', 'race', 'racesec', 'raceter', 'primlang', 'educ']

# Sort by subid and visit date first
yearly_df = yearly_df.sort_values(by=['subid', 'visit_date'])

# Apply forward fill per subid group
yearly_df[demographic_features] = (
    yearly_df.groupby('subid')[demographic_features]
    .ffill()
    .bfill()  # Optional: also backfill in case some subjects have missing values at the start
)

print(f"Number of subjects with yearly survey data: {len(yearly_df['subid'].unique())}")
num_subids_5 = yearly_df['subid'].nunique()
total_rows_5 = len(yearly_df)
log_step(f"Step 5: {num_subids_5} unique subids, {total_rows_5} total rows after forward filling demographic features.")
row_counts_5 = yearly_df['subid'].value_counts().sort_index()
row_counts_5.to_csv(os.path.join(output_path, 'row_counts_step_5.csv'))
missing_5 = yearly_df.isnull().sum()
missing_5[missing_5 > 0].to_csv(os.path.join(output_path, 'missing_counts_step_5.csv'))

#------ Step 6: Check for duplicates ------
log_step("Checking for duplicate (subid, date) rows.")
duplicates = yearly_df.duplicated(subset=['subid', 'visit_date'], keep=False)
dup_rows = yearly_df[duplicates]
if not dup_rows.empty:
    log_step(f"Found {len(dup_rows)} duplicate rows. Saving to 'duplicate_entries.csv'.")
    dup_rows.to_csv(os.path.join(output_path, 'duplicate_entries.csv'), index=False)
else:
    log_step("No duplicate rows found.")

#------ Save cleaned data ------
output_final = os.path.join(output_path, 'yearly_cleaned.csv')
yearly_df.to_csv(output_final, index=False)
log_step(f"Cleaned survey data saved to: {output_final}")

#------ Step 6: Recode age_at_visit, education, and MoCA scores ------
log_step("Step 6: Recoding age buckets, education groups, and MoCA cognitive categories.")

# --- Age Buckets ---
age_bins = list(range(55, 106, 5))  # 55 to 105 in 5-year increments
age_labels = [f"{i}-{i+4}" for i in age_bins[:-1]]
yearly_df['age_bucket'] = pd.cut(yearly_df['age_at_visit'], bins=age_bins, labels=age_labels, right=False)

# --- Education Recoding into 3 groups ---
def recode_education(val):
    if val == 12:
        return "HS"
    elif val in [13, 14, 15]:
        return "Some_College"
    elif val in [16, 17, 18, 19, 20, 21, 22]:
        return "College"
    elif val <= 11:
        return "Below_HS"
    elif val == 99:
        return "Unknown"
    else:
        return "Other"

yearly_df['educ_group'] = yearly_df['educ'].apply(recode_education)

# --- MoCA Cognitive Category Recoding ---
def recode_moca(score):
    if pd.isna(score):
        return "Unknown"
    elif score >= 26:
        return "Normal"
    elif score >= 18:
        return "Mild"
    elif score >= 10:
        return "Moderate"
    else:
        return "Severe"

yearly_df['moca_category'] = yearly_df['mocatots'].apply(recode_moca)

# --- Race Group Recode: White, Non-White, Unknown ---
def label_race(val):
    try:
        val = int(val)
    except:
        return "Unknown"
    
    if val == 1:
        return "White"
    elif val == 99:
        return "Unknown"
    else:
        return "Non-White"

yearly_df['race_group'] = yearly_df['race'].apply(label_race)

print(f"Number of subjects with yearly survey data: {len(yearly_df['subid'].unique())}")
num_subids_6 = yearly_df['subid'].nunique()
total_rows_6 = len(yearly_df)
log_step(f"Step 6: {num_subids_6} unique subids, {total_rows_6} total rows after recoding.")
row_counts_6 = yearly_df['subid'].value_counts().sort_index()
row_counts_6.to_csv(os.path.join(output_path, 'row_counts_step_6.csv'))
missing_6 = yearly_df.isnull().sum()
missing_6[missing_6 > 0].to_csv(os.path.join(output_path, 'missing_counts_step_6.csv'))

# --- Save recoded column counts ---
yearly_df['age_bucket'].value_counts().sort_index().to_csv(os.path.join(output_path, 'age_bucket_counts.csv'))
yearly_df['educ_group'].value_counts().to_csv(os.path.join(output_path, 'educ_group_counts.csv'))
yearly_df['moca_category'].value_counts().to_csv(os.path.join(output_path, 'moca_category_counts.csv'))
yearly_df['race_group'].value_counts().to_csv(os.path.join(output_path, 'race_group_counts.csv'))

# --- Save recoded dataset ---
recoded_output = os.path.join(output_path, 'yearly_recoded.csv')
yearly_df.to_csv(recoded_output, index=False)
log_step(f"Step 6 complete: recoded dataset saved to {recoded_output}")


#------ Final Summary Table ------
summary = pd.DataFrame({
                'unique_subids': [num_subids_0, num_subids_1, num_subids_2, num_subids_3, num_subids_4, num_subids_5, num_subids_6],
                'n_rows':        [total_rows_0, total_rows_1, total_rows_2, total_rows_3, total_rows_4, total_rows_5, total_rows_6]
}, index=['step_0_initial', 'step_1_drop_na','step_2_extracted_data',  'step_3_fill_fall_dates', 'step_4_create_moca_avg', 'step_5_forward_fill_demographics', 'step_6_recode'])

summary.to_csv(os.path.join(output_path, 'summary_pipeline_overview.csv'))
print("Summary saved to summary_pipeline_overview.csv")

#------ Final Missingness Summary ------
missing_all = pd.DataFrame({
    'step_0': missing_0,
    'step_1': missing_1,
    'step_2': missing_2,
    'step_3': missing_3,
    'step_4': missing_4,
    'step_5': missing_5,
    'step_6': missing_6
}).fillna(0).astype(int)

missing_all.to_csv(os.path.join(output_path, 'missing_counts_summary.csv'))
print("Missing value summary saved to missing_counts_summary.csv")