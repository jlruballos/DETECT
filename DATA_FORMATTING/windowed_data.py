import pandas as pd
import numpy as np
import os

# -------- CONFIG --------
csv_path = "/mnt/d/DETECT/OUTPUT/sequence_feature/labeled_daily_data_mean.csv"
window_size = 7 #window size in days
prediction_horizon = 7 #prediction horizon in days 
n_lags =  0 #how many lag features to create
min_subjects = 10 #minimum number of subjects to include in the windowed data
stride = window_size + prediction_horizon #how many days to move the windo forward
output_path = os.path.join('/mnt/d/DETECT', 'OUTPUT', 'windowed_data')

os.makedirs(output_path, exist_ok=True)  # Create output directory if it does not exist
print(f"Output directory created at: {output_path}")

#KEY FEATURES TO LAG
lag_features = ['steps', 'gait_speed', 'awakenings', 'bedexitcount', 'end_sleep_time', 
    'inbed_time', 'outbed_time', 'sleepscore', 
    'durationinsleep', 'durationawake', 'waso', 
    'hrvscore', 'start_sleep_time', 'time_to_sleep', 
    'time_in_bed_after_sleep', 'total_time_in_bed', 'tossnturncount', 
    'sleep_period',  
    'avghr', 'avgrr']

#OTHER CONTEXT FEATURES
static_features = [   
    'birthyr', 'sex', 'hispanic', 'race', 'educ', 'livsitua', 'independ', 'residenc', 'alzdis'
    ]

#LOAD DATA
df = pd.read_csv(csv_path)
df['date'] = pd.to_datetime(df['date']) #parse dates
df = df.sort_values(['subid', 'date']).reset_index(drop=True) 

#check fall data
print("=== FALL ANALYSIS ===")
print(f"Total rows in original data: {len(df)}")
print(f"Total falls in original data: {df['label_fall'].sum()}")
print(f"Unique subjects with falls: {df[df['label_fall']==1]['subid'].nunique()}")
print(f"Fall rate in original data: {df['label_fall'].mean():.3f}")

# Check falls by date range
falls_by_date = df[df['label_fall']==1].groupby('date').size()
print(f"Date range of falls: {falls_by_date.index.min()} to {falls_by_date.index.max()}")
print("========================")

#Auto-detect global start date
subid_counts =df.groupby('date')['subid'].nunique()
valid_dates = subid_counts[subid_counts >= min_subjects]

if valid_dates.empty:
    raise ValueError(f"No valid dates with at least {min_subjects} subjects found.")

global_start_date = valid_dates.index.min()
print(f"Global start date: {global_start_date}")

all_examples = []

for subid in df['subid'].unique():
    sub_df = df[df['subid'] == subid].sort_values('date').reset_index(drop=True) #sort by date
    sub_df = sub_df[sub_df['date'] >= global_start_date].reset_index(drop=True) #filter to valid dates
    
    # Check if there are enough data points for the subject
    max_i = len(sub_df) - window_size - prediction_horizon
    if max_i < 1:
        print(f"Skipping subject {subid} due to insufficient data points.")
        continue
    
    for i in range(0, max_i, stride):
        window = sub_df.iloc[i:i + window_size].copy() #window of data
        future = sub_df.iloc[i + window_size:i + window_size + prediction_horizon]#future data
        
        #skip if missing data
        #if window[lag_features].isna().any().any() or future['label_fall'].isna().any():
         #   continue
        
        #missing_pct = (window[lag_features].isna().sum().sum()) / (len(window) * len(lag_features))
        #if missing_pct > 0.3 or future['label_fall'].isna().any():  # Allow up to 10% missing data
         #   continue
        
        # Fill remaining missing values
        window[lag_features] = window[lag_features].ffill().bfill()
        #--- Add lag features----
        lag_dfs = []
        lag_dfs.append(window) #start with original window
        
        for feat in lag_features:
            lag_data = {}
            for lag in range(1, n_lags + 1):
                lag_data[f"{feat}_lag{lag}"] = window[feat].shift(lag)
            lag_dfs.append(pd.DataFrame(lag_data, index=window.index))
            
        #combine all lagged features
        window = pd.concat(lag_dfs, axis=1)
                
        #remove rows with NaN values after lagging	
        window = window.iloc[n_lags:].reset_index(drop=True)
        #if len(window) == 0:  
         #   print(f"Skipping window for subject {subid} at index {i} due to insufficient data after lagging.")
          #  continue
        #if window.isna().any().any():
         #   print(f"Skipping window for subject {subid} at index {i} due to insufficient data after lagging.")
          #  continue
        
        #print(f"Subject {subid}, Window {i}: Window has {len(window)} usable days after lagging")
        
        #add weekday one-hot encodings
        window['weekday'] = window['date'].dt.weekday
        weekday_dummies = pd.get_dummies(window['weekday'], prefix='wd')
        window = pd.concat([window, weekday_dummies], axis=1)
        
        #compute teh feaure vectors for the window
        feature_vector ={
			'subid': subid,
			'start_date': window['date'].iloc[0],
			'end_date': window['date'].iloc[-1],
			'label': int(future['label_fall'].any())  # label for the prediction horizon
		}
        
        #summarize lag features
        for feat in lag_features:
            feature_vector[f"{feat}_mean"] = window[feat].mean()
            feature_vector[f"{feat}_std"] = window[feat].std()
            feature_vector[f"{feat}_max"] = window[feat].max()
            feature_vector[f"{feat}_min"] = window[feat].min()
            feature_vector[f"{feat}_last"] = window[feat].iloc[-1]
            
            for lag in range(1, n_lags + 1):
                lag_col = f"{feat}_lag{lag}"
                if lag_col in window.columns:  # check if the lag column exists
                    feature_vector[f"{lag_col}_mean"] = window[lag_col].mean()
                    feature_vector[f"{lag_col}_last"] = window[lag_col].iloc[-1]
                
		#add static features (use last available value)
        for sfeat in static_features:
            if sfeat in window.columns: # check if the static feature exists
                feature_vector[sfeat] = window[sfeat].iloc[-1] # use last value in the window
            
        #add time-of-week features: average occurance of each weekday
        for wd_col in [col for col in window.columns if col.startswith('wd_')]:
            feature_vector[f"{wd_col}_mean"] = window[wd_col].mean()
            
        all_examples.append(feature_vector)

print("=== FINAL FALL ANALYSIS ===")
if all_examples:
    result_df = pd.DataFrame(all_examples)
    total_falls_captured = result_df['label'].sum()
    print(f"Falls captured in windowed data: {total_falls_captured}")
    print(f"Fall rate in windowed data: {result_df['label'].mean():.3f}")
    subjects_with_falls = result_df[result_df['label']==1]['subid'].nunique()
    print(f"Unique subjects with falls in windows: {subjects_with_falls}")
print("===============================")
		
# -------- SAVE TO CSV --------
if all_examples:
    result_df = pd.DataFrame(all_examples)
    csv_filename = f"windowed_data_{window_size}_{prediction_horizon}_lag{n_lags}.csv"
    result_df.to_csv(os.path.join(output_path, csv_filename), index=False)
    print(f" Saved {len(result_df)} rows to: {csv_filename}")
    print(f"Unique participants: {result_df['subid'].nunique()}")
else:
    print(" No valid windows generated.")
  