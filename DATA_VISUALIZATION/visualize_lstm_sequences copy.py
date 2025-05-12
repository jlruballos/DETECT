#!/usr/bin/env python3
"""
Visualize Example LSTM Sequences from DETECT Data

Loads generated sequences and plots sample inputs
highlighting differences between positive and negative examples.
"""

__author__ = "Jorge Ruballos"
__email__ = "ruballoj@oregonstate.edu"
__date__ = "2025-4-26"
__version__ = "1.0.0"

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# -------- CONFIG --------
USE_WSL = True  # Set True if running inside WSL/Linux

# Set base path depending on environment
if USE_WSL:
    base_path = '/mnt/d/DETECT'
else:
    base_path = r'D:\DETECT'

# Define path to pre-generated LSTM sequences
sequences_path = os.path.join(base_path, 'OUTPUT', 'generate_lstm_sequences', 'lstm_sequences.npz')

# -------- LOAD DATA --------
print("Loading LSTM sequences...")
# Load saved npz file containing sequences, labels, and participant IDs
data = np.load(sequences_path)
X = data['X']  # Feature sequences (timesteps x features)
y = data['y']  # Corresponding labels for each sequence
subid = data['subid']  # Participant IDs for each sequence

print(f"Loaded X shape: {X.shape}, y shape: {y.shape}, subid shape: {subid.shape}")

# -------- VISUALIZATION CONFIG --------
# List of feature names matching the order in X
features = [
    'gait_speed_norm', 'gait_speed_delta', 'gait_speed_delta_1d', 'gait_speed_ma_7',
    'daily_steps_norm', 'daily_steps_delta', 'daily_steps_delta_1d', 'daily_steps_ma_7',
    'days_since_fall', 'days_until_fall', 'days_since_hospital', 'days_until_hospital'
]

# Find indexes where label == 1 (positive event) and label == 0 (no event)
positive_idx = np.where(y == 1)[0]
negative_idx = np.where(y == 0)[0]

# Randomly select up to 3 positive and 3 negative examples for visualization
np.random.seed(42)
selected_positive_idx = np.random.choice(positive_idx, size=min(3, len(positive_idx)), replace=False)
selected_negative_idx = np.random.choice(negative_idx, size=min(3, len(negative_idx)), replace=False)

# Merge selected examples into a dictionary: label -> sequence index
examples = {f"Positive Example {i+1}": idx for i, idx in enumerate(selected_positive_idx)}
examples.update({f"Negative Example {i+1}": idx for i, idx in enumerate(selected_negative_idx)})

# -------- PLOTTING --------
# Loop through each selected example and plot its features
for title, idx in examples.items():
    sequence = X[idx]  # Get sequence by index

    sns.set(style="whitegrid")  # Use seaborn whitegrid style
    fig, axes = plt.subplots(4, 3, figsize=(16, 12))  # Create a 4x3 grid of plots
    axes = axes.flatten()

    # Plot each feature over the time window (15 timesteps)
    for i in range(len(features)):
        axes[i].plot(range(sequence.shape[0]), sequence[:, i], marker='o')
        # Highlight the final timestep (where prediction is made)
        axes[i].axvline(x=sequence.shape[0]-1, color='red', linestyle='--', linewidth=1.5)
        axes[i].set_title(features[i], fontsize=10)
        axes[i].grid(True)

    # Set overall plot title showing type, label, and participant ID
    plt.suptitle(f"{title} (Label: {y[idx]}, SubID: {subid[idx]})", fontsize=18)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # -------- SAVE FIGURE --------
    # Define output directory for plots
    visualization_output_path = os.path.join(base_path, 'OUTPUT', 'visualize_lstm_sequences')
    os.makedirs(visualization_output_path, exist_ok=True)

    # Save plot as PNG with descriptive filename
    output_filename = f"{title.replace(' ', '_').lower()}_subid_{subid[idx]}_sequence_plot.png"
    plt.savefig(os.path.join(visualization_output_path, output_filename), dpi=300, bbox_inches='tight')

    plt.show()  # Display the plot interactively