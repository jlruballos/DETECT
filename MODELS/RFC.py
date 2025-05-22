import pandas as pd
import numpy as np
import os
import sys
import wandb
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
from sklearn.calibration import calibration_curve
import sklearn

# ---- Setup ----
base_path = '/mnt/d/DETECT'
sys.path.append(os.path.join(base_path, 'HELPERS'))
from helpers import add_future_event_window_labels

csv_path = "/mnt/d/DETECT/OUTPUT/sequence_feature/labeled_daily_data_ffill.csv"
df = pd.read_csv(csv_path)

imputer = "ffill"
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
run_name = f"rf_{imputer}_falls_within7_{timestamp}"

# Add label
df = add_future_event_window_labels(df, event_cols=['label_fall'], horizons=[7])
target_col = 'label_fall_within7'
df = df.dropna(subset=[target_col])
y = df[target_col].astype(int).values
groups = df['subid'].values

# ---- Features ----
features = [
        'gait_speed', 'steps', 'awakenings', 'bedexitcount', 'end_sleep_time',
    'inbed_time', 'outbed_time', 'sleepscore', 'durationinsleep', 'durationawake',
    'waso', 'hrvscore', 'start_sleep_time', 'time_to_sleep', 'time_in_bed_after_sleep',
    'total_time_in_bed', 'tossnturncount', 'sleep_period', 'minhr', 'maxhr',
    'avghr', 'avgrr', 'maxrr', 'minrr', 'steps_norm', 'steps_delta',
    'steps_delta_1d', 'steps_ma_7', 'gait_speed_norm', 'gait_speed_delta',
    'gait_speed_delta_1d', 'gait_speed_ma_7', 'awakenings_norm', 'awakenings_delta',
    'awakenings_delta_1d', 'awakenings_ma_7', 'bedexitcount_norm', 'bedexitcount_delta',
    'bedexitcount_delta_1d', 'bedexitcount_ma_7', 'end_sleep_time_norm',
    'end_sleep_time_delta', 'end_sleep_time_delta_1d', 'end_sleep_time_ma_7',
    'inbed_time_norm', 'inbed_time_delta', 'inbed_time_delta_1d', 'inbed_time_ma_7',
    'outbed_time_norm', 'outbed_time_delta', 'outbed_time_delta_1d', 'outbed_time_ma_7',
    'sleepscore_norm', 'sleepscore_delta', 'sleepscore_delta_1d', 'sleepscore_ma_7',
    'durationinsleep_norm', 'durationinsleep_delta', 'durationinsleep_delta_1d',
    'durationinsleep_ma_7', 'durationawake_norm', 'durationawake_delta',
    'durationawake_delta_1d', 'durationawake_ma_7', 'waso_norm', 'waso_delta',
    'waso_delta_1d', 'waso_ma_7', 'hrvscore_norm', 'hrvscore_delta',
    'hrvscore_delta_1d', 'hrvscore_ma_7', 'start_sleep_time_norm',
    'start_sleep_time_delta', 'start_sleep_time_delta_1d', 'start_sleep_time_ma_7',
    'time_to_sleep_norm', 'time_to_sleep_delta', 'time_to_sleep_delta_1d',
    'time_to_sleep_ma_7', 'time_in_bed_after_sleep_norm', 'time_in_bed_after_sleep_delta',
    'time_in_bed_after_sleep_delta_1d', 'time_in_bed_after_sleep_ma_7',
    'total_time_in_bed_norm', 'total_time_in_bed_delta', 'total_time_in_bed_delta_1d',
    'total_time_in_bed_ma_7', 'tossnturncount_norm', 'tossnturncount_delta',
    'tossnturncount_delta_1d', 'tossnturncount_ma_7', 'sleep_period_norm',
    'sleep_period_delta', 'sleep_period_delta_1d', 'sleep_period_ma_7',
    'minhr_norm', 'minhr_delta', 'minhr_delta_1d', 'minhr_ma_7',
    'maxhr_norm', 'maxhr_delta', 'maxhr_delta_1d', 'maxhr_ma_7',
    'avghr_norm', 'avghr_delta', 'avghr_delta_1d', 'avghr_ma_7',
    'avgrr_norm', 'avgrr_delta', 'avgrr_delta_1d', 'avgrr_ma_7',
    'maxrr_norm', 'maxrr_delta', 'maxrr_delta_1d', 'maxrr_ma_7',
    'minrr_norm', 'minrr_delta', 'minrr_delta_1d', 'minrr_ma_7',
    'days_since_fall'
    ]  # Replace with your actual feature list from previous messages

X = df[features].values

# ---- Group Split ----
splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
train_idx, test_idx = next(splitter.split(X, y, groups=groups))
X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

print("Train size:", len(X_train))
print("Test size:", len(X_test))

# ---- wandb ----
wandb.init(project="detect-rf", name=run_name, config={
    "imputer": imputer,
    "model": "RandomForest",
    "label_shift_days": 7,
    "features": features,
})

# ---- Train Classifier ----
rf = RandomForestClassifier(n_estimators=100, random_state=0)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
y_pred_proba = rf.predict_proba(X_test)[:, 1]

# ---- Metrics ----
metrics = {
    'accuracy': accuracy_score(y_test, y_pred),
    'precision': precision_score(y_test, y_pred),
    'recall': recall_score(y_test, y_pred),
    'f1': f1_score(y_test, y_pred),
    'roc_auc': roc_auc_score(y_test, y_pred_proba)
}
print(metrics)

# ---- Confusion Matrix ----
cm = confusion_matrix(y_test, y_pred)
labels = ["No Fall", "Fall"]
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
wandb.log({"confusion_matrix": wandb.Image(plt)})
plt.close()

# ---- Reliability Diagram ----
bins = np.linspace(0, 1, 11)
bin_centers = (bins[1:] + bins[:-1]) / 2

try:
    # Try to use return_counts if available
    prob_true, prob_pred, counts = calibration_curve(
        y_test, y_pred_proba, n_bins=10, strategy='uniform', return_counts=True
    )
    valid_mask = counts > 0
    bin_centers = bin_centers[valid_mask]
    counts = counts[valid_mask]
    prob_true = prob_true[valid_mask]
    prob_pred = prob_pred[valid_mask]

except TypeError:
    # Fallback if return_counts fails
    prob_true, prob_pred = calibration_curve(
        y_test, y_pred_proba, n_bins=10, strategy='uniform'
    )
    bin_ids = np.digitize(y_pred_proba, bins) - 1
    bin_ids = np.clip(bin_ids, 0, 9)
    counts = np.bincount(bin_ids, minlength=10)
    valid_mask = counts > 0
    counts = counts[valid_mask]
    bin_centers = bin_centers[valid_mask]
    prob_true = prob_true[:len(bin_centers)]
    prob_pred = prob_pred[:len(bin_centers)]

# ---- ECE and Plot ----
ece = np.sum(np.abs(prob_true - bin_centers) * counts / np.sum(counts))

plt.figure(figsize=(6, 6))
plt.bar(bin_centers, prob_true, width=0.1, edgecolor='black', color='blue', label='Outputs', alpha=0.9)
plt.plot([0, 1], [0, 1], '--', color='gray', label='Perfect Calibration')
for x, p, t in zip(bin_centers, prob_pred, prob_true):
    plt.plot([x, x], [p, t], color='red', linewidth=2, alpha=0.5)
plt.text(0.05, 0.05, f'Error={ece*100:.1f}', fontsize=12, bbox=dict(facecolor='lightgray'))
plt.xlabel('Confidence')
plt.ylabel('Accuracy')
plt.title('Reliability Diagram (Bar Style)')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.legend()
plt.grid(True)
wandb.log({"reliability_diagram_bar": wandb.Image(plt)})
plt.close()

# ---- Save Predictions ----
results_df = pd.DataFrame({
    "subid": df.iloc[test_idx]["subid"].values,
    "date": df.iloc[test_idx]["date"].values,
    "true_label": y_test,
    "predicted_label": y_pred,
    "predicted_prob": y_pred_proba
})
results_path = f"/mnt/d/DETECT/OUTPUT/rf/random_forest_predictions_{imputer}_{timestamp}.csv"
os.makedirs(os.path.dirname(results_path), exist_ok=True)
results_df.to_csv(results_path, index=False)
print(f"Saved predictions to: {results_path}")

# ---- Finish ----
wandb.log(metrics)
wandb.log({"ece": ece})
wandb.finish()
