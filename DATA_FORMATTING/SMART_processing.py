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

program_name = 'SMART_processing'

#------ Set up paths ------
base_path = '/mnt/d/DETECT'
output_path = os.path.join(base_path, 'OUTPUT', program_name)
os.makedirs(output_path, exist_ok=True)

#YEARLY_PATH = os.path.join(base_path, 'DETECT_Data', 'Clinical', 'Clinical', 'kaye_365_clin_age_at_visit.csv')
SMART_PATH = os.path.join(base_path, 'DETECT_DATA_080825', 'SMART', 'DETECT_smart_cleaned.csv')

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
log_step("Loading SMART data.")
smart_df = pd.read_csv(SMART_PATH)

#--------RENAME COLUMNS --------
smart_df = smart_df.rename(columns={'Duration (in seconds)': 'duration', #reanme duration column (this is the total time spent in the SMART test)
                                    
                                    'RecordedDate': 'date', #rename recorded date to date
                                    
                                    'Q191_First Click': 'trailsa_practice_first_click', #rename trailsa practice first click time column
									'Q191_Last Click': 'trailsa_practice_last_click', #rename trailsa practice last click time column
									'Q191_Page Submit': 'trailsa_practice_page_submit', #rename trailsa practice page submit time column
									'Q191_Click Count': 'trailsa_practice_click_count', #rename trailsa practice click count column
                                    
                                    'trailsa_timing_First Click': 'trailsa_first_click', #rename trailsa first click time column
									'trailsa_timing_Last Click': 'trailsa_last_click', #rename trailsa last click time column
                                    'trailsa_timing_Page Submit': 'trailsa_page_submit', #rename trailsa page submit time column
                                    'trailsa_timing_Click Count': 'trailsa_click_count', #rename trailsa click count column
                                    
									'trailsb_timing_First Click': 'trailsb_first_click', #rename trailsb first click time column
									'trailsb_timing_Last Click': 'trailsb_last_click', #rename trailsb last click time column
									'trailsb_timing_Page Submit': 'trailsb_page_submit', #rename trailsb page submit time column
									'trailsb_timing_Click Count': 'trailsb_click_count', #rename trailsb click count column
         
									'Q944_First Click': 'trailsb_practice_first_click', #rename trailsb practice first click time column
									'Q944_Last Click': 'trailsb_practice_last_click', #rename trailsb practice last click time column
									'Q944_Page Submit': 'trailsb_practice_page_submit', #rename trailsb practice page submit time column
									'Q944_Click Count': 'trailsb_practice_click_count', #rename trailsb practice click count column
         
									'Q947_First Click': 'stroop_practice_first_click', #rename stroop practice first click time column
									'Q947_Last Click': 'stroop_practice_last_click', #rename stroop practice last click time column
									'Q947_Page Submit': 'stroop_practice_page_submit', #rename stroop practice page submit time column
									'Q947_Click Count': 'stroop_practice_click_count', #rename stroop practice click count column
         
                                    'stroop_timing_First Click': 'stroop_first_click', #rename stroop first click time column
									'stroop_timing_Last Click': 'stroop_last_click', #rename stroop last click time column
									'stroop_timing_Page Submit': 'stroop_page_submit', #rename stroop page submit time column
									'stroop_timing_Click Count': 'stroop_click_count', #rename stro
         
									'DL2_timing_First Click': 'dl2_first_click', #rename dl2 first click time column
									'DL2_timing_Last Click': 'dl2_last_click', #rename dl2 last click time column
									'DL2_timing_Page Submit': 'dl2_page_submit', #rename dl2 page submit time column
									'DL2_timing_Click Count': 'dl2_click_count', #rename dl2 click count column
                                    }) 


# Count before cleaning
num_subids_0 = smart_df['subid'].nunique()
total_rows_0 = len(smart_df)
log_step(f"Step 0: {num_subids_0} unique subids, {total_rows_0} total rows after merge.")
row_counts_0 = smart_df['subid'].value_counts().sort_index()
row_counts_0.to_csv(os.path.join(output_path, 'row_counts_step_0.csv'))
missing_0 = smart_df.isnull().sum()
missing_0[missing_0 > 0].to_csv(os.path.join(output_path, 'missing_counts_step_0.csv'))

#-------parse only date ------
log_step("Parsing month, date and year from date no time")
smart_df['date'] = pd.to_datetime(smart_df['date']).dt.date

#------ Step 1: Drop missing values ------
log_step("Step 1: Dropping rows with missing subid or date.")
smart_df = smart_df.dropna(subset=['subid', 'date'])

num_subids_1 = smart_df['subid'].nunique()
total_rows_1 = len(smart_df)
log_step(f"Step 1: {num_subids_1} unique subids, {total_rows_1} total rows after dropping NAs.")
row_counts_1 = smart_df['subid'].value_counts().sort_index()
row_counts_1.to_csv(os.path.join(output_path, 'row_counts_step_1.csv'))
missing_1 = smart_df.isnull().sum()
missing_1[missing_1 > 0].to_csv(os.path.join(output_path, 'missing_counts_step_1.csv'))

#------ Step 2: Create total_only_test columns ------
log_step("Step 2: Create a total_only_test column subtracting first click from last click for each test.")
# Create total_only_test columns for each test
#practice tests
smart_df['trailsa_practice_only_test'] = smart_df['trailsa_practice_last_click'] - smart_df['trailsa_practice_first_click']
smart_df['trailsb_practice_only_test'] = smart_df['trailsb_practice_last_click'] - smart_df['trailsb_practice_first_click']
smart_df['stroop_practice_only_test'] = smart_df['stroop_practice_last_click'] - smart_df['stroop_practice_first_click']

#actual tests
smart_df['trailsa_only_test'] = smart_df['trailsa_last_click'] - smart_df['trailsa_first_click']
smart_df['trailsb_only_test'] = smart_df['trailsb_last_click'] - smart_df['trailsb_first_click']
smart_df['stroop_only_test'] = smart_df['stroop_last_click'] - smart_df['stroop_first_click']
smart_df['dl2_only_test'] = smart_df['dl2_page_submit']

print(f"Number of subjects with SMART survey data: {len(smart_df['subid'].unique())}")
num_subids_2 = smart_df['subid'].nunique()
total_rows_2 = len(smart_df)
log_step(f"Step 2: {num_subids_2} unique subids, {total_rows_2} total rows after date extraction and converting all date columns.")
row_counts_2 = smart_df['subid'].value_counts().sort_index()
row_counts_2.to_csv(os.path.join(output_path, 'row_counts_step_2.csv'))
missing_2 = smart_df.isnull().sum()
missing_2[missing_2 > 0].to_csv(os.path.join(output_path, 'missing_counts_step_2.csv'))

#------ Step 3: Create total_test_time ------
log_step("Step 3: Create a total_test_time column adding all test durations and total_no_dl2.")
# Create total_test_time column adding all test durations
#total test time with no dl2 
#total test time with practice tests excluded
smart_df['total_test_time'] = smart_df[['trailsa_only_test', 'trailsb_only_test', 'stroop_only_test']].sum(axis=1)
#total practice test time
smart_df['total_practice_test_time'] = smart_df[['trailsa_practice_only_test', 'trailsb_practice_only_test', 'stroop_practice_only_test']].sum(axis=1)
#total test time with practice tests included
smart_df['total_test_time_with_practice'] = smart_df['total_test_time'] + smart_df['total_practice_test_time']

#total test time with dl2
smart_df['total_with_dl2_time'] = smart_df[['trailsa_only_test', 'trailsb_only_test', 'stroop_only_test', 'dl2_only_test']].sum(axis=1) #total time with dl2
smart_df['total_with_dl2_time_with_practice'] = smart_df['total_with_dl2_time'] + smart_df['total_practice_test_time'] #total time with dl2 and practice tests


print(f"Number of subjects with SMART survey data: {len(smart_df['subid'].unique())}")
num_subids_3 = smart_df['subid'].nunique()
total_rows_3 = len(smart_df)
log_step(f"Step 3: {num_subids_3} unique subids, {total_rows_3} total rows after creating total_test_time column.")
row_counts_3 = smart_df['subid'].value_counts().sort_index()
row_counts_3.to_csv(os.path.join(output_path, 'row_counts_step_3.csv'))
missing_3 = smart_df.isnull().sum()
missing_3[missing_3 > 0].to_csv(os.path.join(output_path, 'missing_counts_step_3.csv'))

#------ Step 4: Create total_page_time ------
log_step("Step 4: Create a total_page_time column adding all page durations.")
# Create total_page_time column adding all page durations
smart_df['total_page_time'] = smart_df[['trailsa_page_submit', 'trailsb_page_submit', 'stroop_page_submit', 'dl2_page_submit']].sum(axis=1)

print(f"Number of subjects with SMART survey data: {len(smart_df['subid'].unique())}")
num_subids_4 = smart_df['subid'].nunique()
total_rows_4 = len(smart_df)
log_step(f"Step 4: {num_subids_4} unique subids, {total_rows_4} total rows after creating total_page_time column.")
row_counts_4 = smart_df['subid'].value_counts().sort_index()
row_counts_4.to_csv(os.path.join(output_path, 'row_counts_step_4.csv'))
missing_4 = smart_df.isnull().sum()
missing_4[missing_4 > 0].to_csv(os.path.join(output_path, 'missing_counts_step_4.csv'))


#------ Step 5: Check for duplicates ------
log_step("Checking for duplicate (subid, date) rows.")
duplicates = smart_df.duplicated(subset=['subid', 'date'], keep=False)
dup_rows = smart_df[duplicates]
if not dup_rows.empty:
    log_step(f"Found {len(dup_rows)} duplicate rows. Saving to 'duplicate_entries.csv'.")
    dup_rows.to_csv(os.path.join(output_path, 'duplicate_entries.csv'), index=False)
else:
    log_step("No duplicate rows found.")

#------ Step 11: Drop duplicates  ------
smart_df_copy = smart_df.drop_duplicates(subset=['subid', 'date'], keep='first').copy()

log_step("Checking for duplicate (subid, date) rows.")
duplicates = smart_df_copy.duplicated(subset=['subid', 'date'], keep=False)
dup_rows = smart_df_copy[duplicates]
if not dup_rows.empty:
    log_step(f"Found {len(dup_rows)} duplicate rows after dropping duplicates. Saving to 'duplicate_entries_after_dropping.csv'.")
    dup_rows.to_csv(os.path.join(output_path, 'duplicate_entries_after_dropping.csv'), index=False)
else:
    log_step("No duplicate rows found.")
    
# Log counts after consolidation
num_subids_5 = smart_df_copy['subid'].nunique()
total_rows_5 = len(smart_df_copy)
log_step(f"Step 11: {num_subids_5} unique subids, {total_rows_5} total rows after dropping duplicates to one row per day.")
row_counts_5 = smart_df_copy['subid'].value_counts().sort_index()
row_counts_5.to_csv(os.path.join(output_path, 'row_counts_step_5.csv'))
missing_5 = smart_df_copy.isnull().sum()
missing_5[missing_5 > 0].to_csv(os.path.join(output_path, 'missing_counts_step_5.csv'))

smart_df = smart_df_copy.copy()

#------ Save cleaned data ------
output_final = os.path.join(output_path, 'smart_cleaned.csv')
smart_df.to_csv(output_final, index=False)
log_step(f"Cleaned SMART data saved to: {output_final}")


print(f"Number of subjects with SMART data: {len(smart_df['subid'].unique())}")
num_subids_6 = smart_df['subid'].nunique()
total_rows_6 = len(smart_df)
log_step(f"Step 5: {num_subids_6} unique subids, {total_rows_6} total rows after saving.")
row_counts_6 = smart_df['subid'].value_counts().sort_index()
row_counts_6.to_csv(os.path.join(output_path, 'row_counts_step_6.csv'))
missing_6 = smart_df.isnull().sum()
missing_6[missing_6 > 0].to_csv(os.path.join(output_path, 'missing_counts_step_6.csv'))

#------ Final Summary Table ------
summary = pd.DataFrame({
                'unique_subids': [num_subids_0, num_subids_1, num_subids_2, num_subids_3, num_subids_4, num_subids_5, num_subids_6],
                'n_rows':        [total_rows_0, total_rows_1, total_rows_2, total_rows_3, total_rows_4, total_rows_5, total_rows_6]
}, index=['step_0_initial', 'step_1_drop_na','step_2_only_test',  'step_3_create_total_test_time', 'step_4_create_total_page_time', 'step_5_check_duplicates', 'step_6_after_saving'])

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

# ============================================================
# GRAPHICAL SECTION: one line plot per participant (subid)
# ============================================================

import matplotlib.pyplot as plt
import seaborn as sns

log_step("Step 6: Generating per-participant SMART plots.")

# ---- Ensure output folder exists ----
plot_dir = os.path.join(output_path, "plots_per_subid")
os.makedirs(plot_dir, exist_ok=True)

# ---- Define color palette ----
palette = {
    'Correct': '#2ca02c',     # green
    'Incorrect': '#d62728',   # red
    'incomplete': '#ff7f0e',   # orange
    'incomplete_grid_display_completed': '#ff7f0e',  # orange
    'incomplete_grid_display_loaded': '#ff7f0e',  # orange
}

# ---- Check if DotLocationAnswer exists ----
if 'DotLocationAnswer' not in smart_df.columns:
    log_step("Warning: 'DotLocationAnswer' column not found. Creating placeholder values (all 'Incomplete').")
    smart_df['DotLocationAnswer'] = 'incomplete'

# ---- Sort by date for plotting ----
smart_df = smart_df.sort_values(['subid', 'date'])

# ---- Loop through each participant ----
for subid, sub_df in smart_df.groupby('subid'):
    plt.figure(figsize=(7, 4))
    sub_df = sub_df.sort_values('date')

    # ---- Line 1: total_test_time (gray solid) ----
    sns.lineplot(
        data=sub_df,
        x='date', y='total_test_time',
        color='gray', linewidth=1.5,
        label='Total Test Time (no DL2)'
    )

    # ---- Line 2: total_with_dl2_time (blue dashed) ----
    sns.lineplot(
        data=sub_df,
        x='date', y='total_with_dl2_time',
        color='blue', linestyle='--', linewidth=1.5,
        label='Total Test Time (with DL2)'
    )
    
    #---- Line 3: total_test_time_with_practice (purple dotted) ----
    sns.lineplot(
        data=sub_df,
        x='date', y='total_test_time_with_practice',
        color='purple', linestyle=':', linewidth=1.5,
        label='Total Test Time (with Practice)'
    )
    #---- Line 4: total_with_dl2_time_with_practice (brown dash-dot) ----
    sns.lineplot(
        data=sub_df,
        x='date', y='total_with_dl2_time_with_practice',
        color='brown', linestyle='-.', linewidth=1.5,
        label='Total Test Time (with DL2 and Practice)'
    )
    
    #---- Line 5: total_practice_test_time (green solid) ----
    sns.lineplot(
        data=sub_df,
        x='date', y='total_practice_test_time',
        color='green', linestyle='-', linewidth=1.5,
        label='Total Practice Test Time'
    )

    # ---- Colored dots by correctness ----
    sns.scatterplot(
        data=sub_df,
        x='date', y='total_with_dl2_time',
        hue='DotLocationAnswer',
        palette=palette,
        s=60, edgecolor='black'
    )

    # ---- Title and axis labels ----
    plt.title(f'Subject {subid} — Total Test Time Over Time', fontsize=12)
    plt.xlabel('Date')
    plt.ylabel('Total Test Time (seconds)')

    # ---- Move legend inside and make it smaller ----
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(
        handles=handles, labels=labels,
        title='',
        fontsize=8,               # smaller text
        loc='upper right',        # inside top-right corner
        frameon=True,
        framealpha=0.8,
        fancybox=True,
        borderpad=0.4
    )

    plt.tight_layout(pad=1.5)

    # ---- Save figure ----
    fig_path = os.path.join(plot_dir, f"subid_{subid}.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()


log_step(f"Step 6 complete: Saved per-participant plots to {plot_dir}")


