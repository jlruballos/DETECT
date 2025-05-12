#!/usr/bin/env python3
"""
Generate LSTM Sequences from Engineered DETECT Data

Loads labeled daily data and creates sliding window sequences
for LSTM model training, with a 7-day future prediction window.
"""

__author__ = "Jorge Ruballos"
__email__ = "ruballoj@oregonstate.edu"
__date__ = "2025-4-26"
__version__ = "1.0.0"

import pandas as pd
import numpy as np
import os
import sys
# -------- CONFIG --------
USE_WSL = True

if USE_WSL:
    base_path = '/mnt/d/DETECT'
else:
    base_path = r'D:\DETECT'

# Add helpers directory to system path
sys.path.append(os.path.join(base_path, 'HELPERS'))
from helpers import get_label

program_name = 'generate_lstm_sequences'
output_path = os.path.join(base_path, 'OUTPUT', program_name)
os.makedirs(output_path, exist_ok=True)
print(f"Output directory created at: {output_path}")

# Load engineered daily data
print("Loading labeled daily data...")
daily_data_path = os.path.join(base_path, 'OUTPUT', 'sequence_feature', 'labeled_daily_data_vae.csv')

daily_data = pd.read_csv(daily_data_path)

# Extract imputation method from filename (e.g., "vae" from "labeled_daily_data_vae.csv")
imputation_method = os.path.basename(daily_data_path).replace('labeled_daily_data_', '').replace('.csv', '')

# -------- SEQUENCE GENERATION CONFIG --------
timesteps = 7  # Number of past days for each input sequence
label_shift = 7  # Predict 7 days after the end of each window

#teh features below inlcuded blanks so they can not be used in the model 
#'days_since_fall', 'days_until_fall',  'gait_speed_delta_1d', 'days_since_hospital', 'days_until_hospital', 'daily_steps_delta_1d',

# Features to include in each sequence
features = [
    'gait_speed_norm', 'gait_speed_delta', 'gait_speed_ma_7',
    'daily_steps_norm', 'daily_steps_delta', 'daily_steps_ma_7', 
    'gait_speed', 'daily_steps', 'gait_speed_mask', 'daily_steps_mask',   
]

X_all = []
y_all = []

# -------- GENERATE SEQUENCES --------
subject_ids = daily_data['subid'].unique()
print(f"Generating sequences for {len(subject_ids)} subjects...")

subid_all = []

for subid in subject_ids:
    sub_df = daily_data[daily_data['subid'] == subid].sort_values('date').reset_index(drop=True)

    # Drop rows with missing feature values
    #sub_df = sub_df.dropna(subset=features)
    
    total_rows = len(sub_df)
    print(f"\nSubject {subid}: {total_rows} rows before windowing")

    valid_sequences = 0

    for i in range(len(sub_df) - timesteps - label_shift):
        window = sub_df.iloc[i:i + timesteps]
        label = get_label(sub_df, i, timesteps, label_shift=label_shift)

        # if window[features].isnull().values.any() or pd.isnull(label):
        #     continue
        has_nan = window[features].isnull().values.any()
        label_nan = pd.isnull(label)
        if has_nan or label_nan:
            if has_nan:
               print(f"  Skipping window {i}-{i+timesteps} (NaNs in features)")
            if label_nan:
               print(f"  Skipping window {i}-{i+timesteps} (NaN label)")
            continue
        
        valid_sequences += 1
        X_all.append(window[features].values)
        y_all.append(label)
        subid_all.append(subid) # Store subid for each sequence
        print(f"Subject {subid}: {valid_sequences} valid sequences generated")

# Convert to arrays
X = np.array(X_all)
y = np.array(y_all)

# -------- SAVE OUTPUT --------
print("Saving LSTM sequences...")

# Save npz directly in the program-named output folder
npz_filename = f"lstm_sequences_{imputation_method}.npz"
np.savez(os.path.join(output_path, npz_filename), X=X, y=y, subid=np.array(subid_all))

print("Saved LSTM sequences.")
print(f"X shape: {X.shape}, y shape: {y.shape}")
print("Saved LSTM sequences.")
print(f"X shape: {X.shape}, y shape: {y.shape}")
print(f"number of subid: {len(np.unique(subid_all))}")
print(f"number of sequences: {len(X_all)}")