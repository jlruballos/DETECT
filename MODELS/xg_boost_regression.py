import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import wandb
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns

# Define base paths depending on environment
base_path = '/mnt/d/DETECT'

# Add helpers directory to system path
sys.path.append(os.path.join(base_path, 'HELPERS'))
from helpers import add_future_event_window_labels

# Load data
csv_path = "/mnt/d/DETECT/OUTPUT/sequence_feature/labeled_daily_data_vae.csv"
df = pd.read_csv(csv_path)
imputer = "vae"

print(df.columns)

# Add binary future label
df = add_future_event_window_labels(df, event_cols=['label_fall'], horizons=[7])

# Define feature columns
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
]

target_col = 'days_until_fall'

# Drop rows where the target is missing
df = df.dropna(subset=[target_col])

# Prepare data
X = df[features].values
y = df[target_col].astype(int).values
groups = df['subid'].values

# Train-test split using group shuffle
splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
train_idx, test_idx = next(splitter.split(X, y, groups=groups))
X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

# Start wandb logging
wandb.init(project="detect-xgboost",
           name="xgb_days_until_fall",
           config={
               "imputer": imputer,
               "model": "XGBoost",
               "target": target_col,
               "features": features,
           })
from wandb.integration.xgboost import WandbCallback
callbacks = [WandbCallback()]

# Check for leakage
train_subids = set(groups[train_idx])
test_subids = set(groups[test_idx])
if train_subids.intersection(test_subids):
    print("⚠️ Data leakage detected!")
else:
    print("✅ No overlap between train and test sets.")

# Train regression model
dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=features)
dtest = xgb.DMatrix(X_test, label=y_test, feature_names=features)

params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'random_state': 0,
}

bst = xgb.train(
    params=params,
    dtrain=dtrain,
    num_boost_round=200,
    evals=[(dtrain, "train"), (dtest, "eval")],
    callbacks=callbacks
)

# Predictions
y_pred = bst.predict(dtest)

# Evaluation metrics
metrics = {
    'mae': mean_absolute_error(y_test, y_pred),
    'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
    'r2': r2_score(y_test, y_pred),
}
print(metrics)

# Plot prediction vs. true values
plt.figure(figsize=(6, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.3)
plt.xlabel("True Days Until Fall")
plt.ylabel("Predicted Days Until Fall")
plt.title("Prediction vs. Ground Truth")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')  # identity line
wandb.log({"prediction_vs_truth": wandb.Image(plt)})
plt.close()

# Save predictions
results_df = pd.DataFrame({
    "subid": df.iloc[test_idx]["subid"].values,
    "date": df.iloc[test_idx]["date"].values,
    "true_value": y_test,
    "predicted_value": y_pred
})
results_path = "/mnt/d/DETECT/OUTPUT/xg_boost/xgboost_predictions.csv"
os.makedirs(os.path.dirname(results_path), exist_ok=True)
results_df.to_csv(results_path, index=False)
print(f"Saved predictions to: {results_path}")

# Log metrics
wandb.log(metrics)
wandb.finish()