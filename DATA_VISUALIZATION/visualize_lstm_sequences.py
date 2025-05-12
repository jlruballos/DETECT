import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# -------- CONFIG --------
USE_WSL = True
if USE_WSL:
    base_path = '/mnt/d/DETECT'
else:
    base_path = r'D:\DETECT'

sequences_path = os.path.join(base_path, 'OUTPUT', 'generate_lstm_sequences', 'lstm_sequences.npz')
visualization_output_path = os.path.join(base_path, 'OUTPUT', 'visualize_lstm_sequences')

# Load data
print("Loading LSTM sequences...")
data = np.load(sequences_path)
X = data['X']
y = data['y']
subid = data['subid']

print(f"Loaded X shape: {X.shape}, y shape: {y.shape}, subid shape: {subid.shape}")

# Feature names
features = [
    'gait_speed_norm', 'gait_speed_delta', 'gait_speed_delta_1d', 'gait_speed_ma_7',
    'daily_steps_norm', 'daily_steps_delta', 'daily_steps_delta_1d', 'daily_steps_ma_7',
    'days_since_fall', 'days_until_fall', 'days_since_hospital', 'days_until_hospital'
]

# Create plots per sequence
for i in range(len(X)):
    sequence = X[i]
    label = y[i]
    sid = subid[i]

    sns.set(style="whitegrid")
    fig, axes = plt.subplots(4, 3, figsize=(14, 10))
    axes = axes.flatten()

    for f_idx in range(len(features)):
        axes[f_idx].plot(range(sequence.shape[0]), sequence[:, f_idx], marker='o')
        axes[f_idx].axvline(x=sequence.shape[0] - 1, color='red', linestyle='--', linewidth=1.2)
        axes[f_idx].set_title(features[f_idx], fontsize=10)
        axes[f_idx].grid(True)

    plt.suptitle(f'Subject {sid} - Sequence {i + 1} (Label: {label})', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    subfolder = os.path.join(visualization_output_path, f'subid_{sid}')
    os.makedirs(subfolder, exist_ok=True)
    filename = f'subject_{sid}_seq_{i + 1}_label_{label}.png'
    plt.savefig(os.path.join(subfolder, filename), dpi=300, bbox_inches='tight')
    plt.close()

print("All individual sequence plots saved per subject.")
