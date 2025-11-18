import pandas as pd
import os
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

program_name = 'fall_gait_sp_plots'

#------ Set up paths ------
base_path = '/mnt/d/DETECT'
base_c_path = '/mnt/d/DETECT_33125/DETECT_Data_Pull_2024-12-16'
output_dir = os.path.join(base_path, 'OUTPUT', program_name)
os.makedirs(output_dir, exist_ok=True)

#YEARLY_PATH = os.path.join(base_path, 'DETECT_Data', 'Clinical', 'Clinical', 'kaye_365_clin_age_at_visit.csv')
GAIT_PATH = os.path.join(base_path, 'OUTPUT', 'GAIT', 'COMBINED_NYCE_Area_Data_DETECT_GAIT_summary.csv')
CONTEXT_PATH = os.path.join(base_c_path, '_CONTEXT_FILES', 'Study_Home-Subject_Dates_2024-12-16', 'homeids_subids_NYCE.csv')
FALLS_PATH = os.path.join(base_path, 'OUTPUT', 'survey_processing', 'survey_cleaned.csv')
STEPS_PATH = os.path.join(base_path, 'OUTPUT', 'watch_steps_processing', 'watch_steps_cleaned.csv')

# Load and process gait data
gait_data = pd.read_csv(GAIT_PATH, low_memory=False)
mapping_data = pd.read_csv(CONTEXT_PATH, low_memory=False)
# Read fall dates
falls = pd.read_csv(FALLS_PATH, low_memory=False)
steps_df = pd.read_csv(STEPS_PATH, low_memory=False)

# Remove homeids that have multiple subids
mapping_data = mapping_data.groupby('home_id').filter(lambda x: len(x) == 1) 
print("Mapping data head:")
print(mapping_data.head())

gait_data = pd.merge(gait_data, mapping_data, left_on='homeid', right_on='home_id', how='inner')
gait_data = gait_data.drop(columns=['home_id'])
gait_data = gait_data.rename(columns={'sub_id': 'subid'})

# Define functions for rounding
def nearest_fifty(series):
    return (np.round(series / 50) * 50).astype(int)

def nearest_lower_fifty(series):
    return (series // 50) * 50

# Function to remove outliers using IQR method
def remove_outliers(df, column_name):
    Q1 = df[column_name].quantile(0.25)
    Q3 = df[column_name].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column_name] >= lower_bound) & (df[column_name] <= upper_bound)]

# Load and process steps data
#root = '/mnt/d/DETECT/OUTPUT/watch_steps_processing/watch_steps_cleaned.csv'
#os.chdir(root)

#file_names = [file for file in os.listdir() if file.endswith('.csv')]
#steps_df = pd.concat((pd.read_csv(file) for file in file_names), ignore_index=True)


falls['cutoff_dates'] = pd.to_datetime(falls['fall1_date']).dt.date
falls['injury'] = falls['FALL1_INJ']
falls['hospital_visit'] = falls['HCRU1_DATE']
falls['change_living'] = falls['SPACE_DATE']
falls['med_change'] = falls['MED1_DATE']

print("Falls data head:")
print(falls.head())

# Create output directory for saved plots
#output_dir = r'D:\DETECT\OUTPUT\gait_steps_fall_plots'
#os.makedirs(output_dir, exist_ok=True)

# Get unique subjects that have both gait and steps data
unique_subjects = list(set(gait_data['subid']).intersection(set(steps_df['subid'])))
print(f"Processing {len(unique_subjects)} subjects")

# Process each subject
for subid in unique_subjects:
    print(f"Processing subject {subid}")
    
    # Filter data for current subject
    gait_data_sub = gait_data[gait_data['subid'] == subid].copy()
    if len(gait_data_sub) == 0:
        print(f"No gait data for subject {subid}, skipping")
        continue
        
    gait_data_sub['date'] = pd.to_datetime(gait_data_sub['start_time']).dt.date

    # Calculate daily average gait speed
    gait_avg_df = gait_data_sub.groupby('date')['gait_speed'].mean().reset_index()
    if len(gait_avg_df) == 0:
        print(f"No valid gait speed data for subject {subid}, skipping")
        continue
    
    # Process steps data
    sub_steps = steps_df[steps_df['subid'] == subid].copy()
    if len(sub_steps) == 0:
        print(f"No steps data for subject {subid}, skipping")
        continue
        
    sub_steps['date'] = pd.to_datetime(sub_steps['date']).dt.date
    #sub_steps['first'] = pd.to_datetime(sub_steps['first'], utc=True)
    #sub_steps['last'] = pd.to_datetime(sub_steps['last'], utc=True)
    sub_steps['daily_steps'] = sub_steps['steps']

    # Remove invalid data
    sub_steps = sub_steps[sub_steps['daily_steps'].notna() & (sub_steps['daily_steps'] > 0)]
    if len(sub_steps) == 0:
        print(f"No valid steps data for subject {subid}, skipping")
        continue
    
    # Get fall dates for this subject
    subject_falls = falls[falls['subid'] == subid].copy()
    cutoff_dates = subject_falls['cutoff_dates'].dropna()
    
    if len(cutoff_dates) == 0:
        print(f"No fall data for subject {subid}, skipping")
        continue
        
    if len(cutoff_dates) > 1:
        cutoff_date = cutoff_dates.min()
    else:
        cutoff_date = cutoff_dates.iloc[0]

    # Calculate days since the first fall event for steps data
    sub_steps['Days Since'] = [(d - cutoff_date).days for d in sub_steps['date']]
    
    #outlier removal
    gait_avg_df = remove_outliers(gait_avg_df, 'gait_speed')
    sub_steps = remove_outliers(sub_steps, 'daily_steps')
    
    # Calculate 7-day moving average for steps
    window_size = 1
    sub_steps['Moving Average Steps'] = sub_steps['daily_steps'].rolling(window=window_size).mean()

    # Align the gait speed data with the same timeline
    gait_avg_df['Days Since'] = [(d - cutoff_date).days for d in gait_avg_df['date']]

    # Create a 7-day moving average for gait speed as well
    gait_avg_df['Moving Average Gait'] = gait_avg_df['gait_speed'].rolling(window=window_size).mean()

    try:
        # Create visualization with separate subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

        # Plot steps data in top subplot
        ax1.plot(sub_steps['Days Since'], sub_steps['daily_steps'], color='lightblue', marker='.', linestyle='', alpha=0.5, label='Daily Steps')
        ax1.plot(sub_steps['Days Since'], sub_steps['Moving Average Steps'], color='blue', linestyle='-', linewidth=2, label='7-day Avg Steps')
        ax1.set_ylabel('Daily Steps')
        ax1.set_title(f'Subject {subid}: Daily Steps Since First Fall')
        ax1.grid(True, alpha=0.3)

        # Plot gait speed data in bottom subplot
        ax2.plot(gait_avg_df['Days Since'], gait_avg_df['gait_speed'], color='lightcoral', marker='.', linestyle='', alpha=0.5, label='Daily Gait Speed')
        ax2.plot(gait_avg_df['Days Since'], gait_avg_df['Moving Average Gait'], color='red', linestyle='-', linewidth=2, label='7-day Avg Gait Speed')
        ax2.set_ylabel('Gait Speed (m/s)')
        ax2.set_xlabel('Days Since First Fall Event')
        ax2.set_title('Gait Speed Since First Fall')
        ax2.grid(True, alpha=0.3)

        # Plot fall events on both subplots
        fall_plotted = False
        injury_plotted = False
        for _, row in subject_falls.iterrows():
            date = row['cutoff_dates']
            if pd.notna(date):
                days_since = (date - cutoff_date).days
                if pd.notna(row['injury']) and row['injury'] == 1.0:
                    label = 'Reported Injury' if not injury_plotted else None
                    ax1.axvline(x=days_since, color='orange', linestyle='-', linewidth=2, label=label)
                    ax2.axvline(x=days_since, color='orange', linestyle='-', linewidth=2)
                    injury_plotted = True
                else:
                    label = 'Fall (No Injury)' if not fall_plotted else None
                    ax1.axvline(x=days_since, color='grey', linestyle='-', label=label)
                    ax2.axvline(x=days_since, color='grey', linestyle='-')
                    fall_plotted = True

        # Set custom x-axis range
        min_days = min(sub_steps['Days Since'].min(), gait_avg_df['Days Since'].min())
        max_days = max(sub_steps['Days Since'].max(), gait_avg_df['Days Since'].max())
        
        # Handle cases where min_days or max_days might be NaN
        if pd.notna(min_days) and pd.notna(max_days):
            tick_range = range(nearest_lower_fifty(min_days), nearest_fifty(max_days) + 50, 50)
            ax2.set_xticks(ticks=tick_range)
            ax2.set_xticklabels(labels=tick_range)
        
        # Add legends
        ax1.legend(loc='best')
        ax2.legend(loc='best')

        plt.tight_layout()
        
        # Save the figure
        output_path = os.path.join(output_dir, f'subject_{subid}_falls_analysis.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved plot for subject {subid} to {output_path}")
        
    except Exception as e:
        print(f"Error plotting data for subject {subid}: {e}")
        continue

print("Processing complete. All graphs saved to:", output_dir)