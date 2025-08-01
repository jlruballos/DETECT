#!/usr/bin/env python3
"""
Clinical Participant Processing Pipeline

This script processes clinical enrollment data from the DETECT study to:
- Recode amyloid status to binary
- Drop rows with missing key values
- Detect and save duplicates
- Add year column for stratification
- Generate demographic bar plots over time (counts & proportions)
- Log per-step row counts and missingness summaries

Outputs:
- Cleaned clinical dataset (`clinical_cleaned.csv`)
- Per-step row counts and missingness
- Stacked and side-by-side barplots (counts and proportions)

Author: Jorge Ruballos
Email: ruballoj@oregonstate.edu
Date: 2025-07-31
Version: 1.0.0
"""

import pandas as pd
import os
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.api.types import is_numeric_dtype

# ---------- CONFIG ----------
base_path = '/mnt/d/DETECT'
input_path = os.path.join(base_path, 'DETECT_Data', 'Clinical', 'Clinical', 'DETECT-AD_Enrolled_Amyloid Status_PET_SUVR_QUEST_CENTILOID_20250116.xlsx')
output_path = os.path.join(base_path, 'OUTPUT', 'clinical_participant_processing')
os.makedirs(output_path, exist_ok=True)

# ---------- LOGGING ----------
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
logfile = os.path.join(output_path, f"pipeline_log_{timestamp}.txt")
def log_step(message):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_msg = f"[{ts}] {message}"
    print(full_msg)
    with open(logfile, "a") as f:
        f.write(full_msg + "\n")

# ---------- Step 0: Load and Recode ----------
log_step("Step 0: Loading clinical participant data.")
df = pd.read_excel(input_path)
df = df.rename(columns={'sub_id': 'subid'})
df['subid'] = df['subid'].astype(str)
df['amyloid'] = df['clinical amyloid (+/-) read'].replace({'Positive': 1, 'Negative': 0})

num_subids_0 = df['subid'].nunique()
total_rows_0 = len(df)
log_step(f"Step 0: {num_subids_0} unique subids, {total_rows_0} total rows.")
df['subid'].value_counts().sort_index().to_csv(os.path.join(output_path, 'row_counts_step_0.csv'))
missing_0 = df.isnull().sum()
missing_0[missing_0 > 0].to_csv(os.path.join(output_path, 'missing_counts_step_0.csv'))

# ---------- Step 1: Drop NAs ----------
log_step("Step 1: Dropping rows with missing subid or amyloid status.")
df_clean = df.dropna(subset=['subid', 'amyloid'])

num_subids_1 = df_clean['subid'].nunique()
total_rows_1 = len(df_clean)
log_step(f"Step 1: {num_subids_1} unique subids, {total_rows_1} total rows after dropping NAs.")
df_clean['subid'].value_counts().sort_index().to_csv(os.path.join(output_path, 'row_counts_step_1.csv'))
missing_1 = df_clean.isnull().sum()
missing_1[missing_1 > 0].to_csv(os.path.join(output_path, 'missing_counts_step_1.csv'))

# ---------- Step 2: Check for Duplicates ----------
log_step("Step 2: Checking for duplicates.")
duplicates = df_clean.duplicated(subset=['subid'], keep=False)
dup_rows = df_clean[duplicates]
if not dup_rows.empty:
    log_step(f"Found {len(dup_rows)} duplicate rows. Saving to 'duplicate_entries.csv'.")
    dup_rows.to_csv(os.path.join(output_path, 'duplicate_entries.csv'), index=False)
else:
    log_step("No duplicate rows found.")

num_subids_2 = df_clean['subid'].nunique()
total_rows_2 = len(df_clean)
log_step(f"Step 2: {num_subids_2} unique subids, {total_rows_2} total rows after duplicate check.")
df_clean['subid'].value_counts().sort_index().to_csv(os.path.join(output_path, 'row_counts_step_2.csv'))
missing_2 = df_clean.isnull().sum()
missing_2[missing_2 > 0].to_csv(os.path.join(output_path, 'missing_counts_step_2.csv'))

# ---------- Step 3: Save Cleaned ----------
log_step("Step 3: Saving cleaned dataset.")
output_csv = os.path.join(output_path, 'clinical_cleaned.csv')
df_clean.to_csv(output_csv, index=False)

num_subids_3 = df_clean['subid'].nunique()
total_rows_3 = len(df_clean)
log_step(f"Step 3: {num_subids_3} unique subids, {total_rows_3} total rows after saving.")
df_clean['subid'].value_counts().sort_index().to_csv(os.path.join(output_path, 'row_counts_step_3.csv'))
missing_3 = df_clean.isnull().sum()
missing_3[missing_3 > 0].to_csv(os.path.join(output_path, 'missing_counts_step_3.csv'))

# ---------- Step 4: Barplots ----------
log_step("Step 4: Creating amyloid barplots.")
plot_dir = os.path.join(output_path, 'barplots')
os.makedirs(plot_dir, exist_ok=True)

# Compute proportions manually
df_plot = df_clean.copy()
df_plot['amyloid'] = df_plot['amyloid'].replace({1: 'Positive', 0: 'Negative'})
prop_df = df_plot['amyloid'].value_counts().reset_index()
prop_df.columns = ['amyloid', 'count']
total = prop_df['count'].sum()
prop_df['percent'] = 100 * prop_df['count'] / total

# Barplot - Count
fig1, ax1 = plt.subplots(figsize=(6, 5))
sns.barplot(data=prop_df, x='amyloid', y='count', palette='Set2', ax=ax1)
for i, row in prop_df.iterrows():
    ax1.text(i, row['count'] + 1, f"{int(row['count'])}", ha='center', va='bottom', fontsize=10)
ax1.set_title("Amyloid Status (Counts)")
ax1.set_ylabel("Count")
plt.tight_layout()
fig1.savefig(os.path.join(plot_dir, 'amyloid_counts.png'))
plt.close(fig1)

# Barplot - Proportion
fig2, ax2 = plt.subplots(figsize=(6, 5))
sns.barplot(data=prop_df, x='amyloid', y='percent', palette='Set2', ax=ax2)
for i, row in prop_df.iterrows():
    ax2.text(i, row['percent'] + 1, f"{row['percent']:.1f}%", ha='center', va='bottom', fontsize=10)
ax2.set_title("Amyloid Status (Proportions)")
ax2.set_ylabel("Percentage")
plt.tight_layout()
fig2.savefig(os.path.join(plot_dir, 'amyloid_props.png'))
plt.close(fig2)

# ---------- Step 5: Summary ----------
summary = pd.DataFrame({
    'unique_subids': [num_subids_0, num_subids_1, num_subids_2, num_subids_3],
    'n_rows':        [total_rows_0, total_rows_1, total_rows_2, total_rows_3]
}, index=['step_0_initial', 'step_1_drop_na', 'step_2_duplicate_check', 'step_3_save_cleaned'])
summary.to_csv(os.path.join(output_path, 'summary_pipeline_overview.csv'))
print("Summary saved to summary_pipeline_overview.csv")

missing_all = pd.DataFrame({
    'step_0': missing_0,
    'step_1': missing_1,
    'step_2': missing_2,
    'step_3': missing_3
}).fillna(0).astype(int)
missing_all.to_csv(os.path.join(output_path, 'missing_counts_summary.csv'))
print("Missing value summary saved to missing_counts_summary.csv")
