#!/usr/bin/env python3
"""
Global VAE Imputation Pipeline for DETECT Sensor Data

This script aggregates all subjects' time series data, trains a global corruption-aware VAE,
and imputes missing values across the full dataset using the trained model.

The imputed data is split back per subject and saved with full feature engineering.
"""

import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime
import torch
import matplotlib.pyplot as plt

# -------- CONFIG --------
USE_WSL = True
VAE_EPOCHS = 30
VAE_VARIANT = 'Encoder + Decoder Mask'
GENERATE_HISTO = True
GENERATE_LINE = True

if USE_WSL:
    base_path = '/mnt/d/DETECT'
else:
    base_path = 'D:/DETECT'

# Add helpers directory to system path
sys.path.append(os.path.join(base_path, 'HELPERS'))

from vae_imputer_global import (
    prepare_global_dataloader,
    train_global_vae,
    impute_with_trained_vae
)

program_name = 'global_vae_feature'
output_path = os.path.join(base_path, 'OUTPUT', program_name)
os.makedirs(output_path, exist_ok=True)

# -------- PATHS --------
STACKED_DATA_PATH = os.path.join(base_path, 'OUTPUT', 'raw_export_for_r', 'raw_daily_data_all_subjects.csv')

# -------- LOAD --------
print("Loading stacked data for global VAE...")
df = pd.read_csv(STACKED_DATA_PATH)
df['date'] = pd.to_datetime(df['date'])

input_columns = [
    'steps', 'gait_speed', 'awakenings', 'bedexitcount', 'end_sleep_time', 
    'inbed_time', 'outbed_time', 'sleepscore', 'durationinsleep', 'durationawake', 'waso',
    'hrvscore', 'start_sleep_time', 'time_to_sleep', 'time_in_bed_after_sleep', 'total_time_in_bed',
    'tossnturncount', 'sleep_period', 'minhr', 'maxhr', 'avghr', 'avgrr', 'maxrr', 'minrr'
]

# -------- STACK ALL SUBJECTS --------
print("Preparing data and masks for VAE training...")
subject_ids = df['subid'].unique()
all_data, all_masks = [], []
subid_lengths = {}

for subid in subject_ids:
    sub_df = df[df['subid'] == subid]
    data = sub_df[input_columns].values.astype(np.float32)
    mask = ~np.isnan(data)
    if data.shape[0] == 0:
        continue
    all_data.append(data)
    all_masks.append(mask.astype(np.float32))
    subid_lengths[subid] = data.shape[0]

# -------- TRAIN GLOBAL VAE --------
dataloader, scaler = prepare_global_dataloader(all_data, all_masks)
model = train_global_vae(dataloader, input_dim=len(input_columns), variant=VAE_VARIANT, epochs=VAE_EPOCHS)

# -------- IMPUTE --------
imputed_frames = []
start = 0
for subid in subject_ids:
    length = subid_lengths[subid]
    end = start + length
    sub_df = df[df['subid'] == subid].copy()
    imputed_df = impute_with_trained_vae(sub_df.copy(), input_columns=input_columns, model=model, scaler=scaler)

    if GENERATE_HISTO or GENERATE_LINE:
        sub_dir = os.path.join(output_path, 'diagnostics')
        histo_dir = os.path.join(sub_dir, 'histograms', f'subid_{subid}')
        line_dir = os.path.join(sub_dir, 'lineplots', f'subid_{subid}')
        os.makedirs(histo_dir, exist_ok=True)
        os.makedirs(line_dir, exist_ok=True)

        for col in input_columns:
            orig_vals = sub_df[col]
            imp_vals = imputed_df[col]
            if GENERATE_HISTO:
                plt.figure()
                plt.hist(orig_vals.dropna(), bins=30, alpha=0.5, label='Original')
                plt.hist(imp_vals[orig_vals.isna()], bins=30, alpha=0.5, label='Imputed')
                plt.legend()
                plt.title(f"Histogram: {col} - Subject {subid}")
                plt.savefig(os.path.join(histo_dir, f"{col}_hist_overlay_{subid}.png"))
                plt.close()
            if GENERATE_LINE:
                plt.figure()
                plt.plot(imputed_df['date'], imp_vals, label='Imputed', color='black')
                plt.scatter(imputed_df['date'][orig_vals.isna()], imp_vals[orig_vals.isna()], color='red', label='Imputed Points')
                plt.title(f"Imputation for {col} - Subject {subid}")
                plt.legend()
                plt.savefig(os.path.join(line_dir, f"{col}_{subid}.png"))
                plt.close()

    imputed_frames.append(imputed_df)
    start = end

final_df = pd.concat(imputed_frames)

# -------- SAVE OUTPUT --------
final_df.to_csv(os.path.join(output_path, f"labeled_daily_data_global_vae.csv"), index=False)
print(f"Saved global VAE-imputed dataset to {output_path}")
