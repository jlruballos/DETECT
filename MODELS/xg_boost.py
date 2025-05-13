import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import wandb
import os
import sys
from sklearn.model_selection import GroupShuffleSplit
import matplotlib.pyplot as plt
import seaborn as sns

# Define base paths depending on environment
base_path = '/mnt/d/DETECT'

# Add helpers directory to system path
sys.path.append(os.path.join(base_path, 'HELPERS'))
from helpers import add_future_event_window_labels

csv_path = "/mnt/d/DETECT/OUTPUT/sequence_feature/labeled_daily_data_vae.csv"

df = pd.read_csv(csv_path)

print(df.columns)

#add label for "fall within next 7 days"
df = add_future_event_window_labels(df, event_cols=['label_fall'], horizons=[7])

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
    'days_since_fall', 'days_until_fall', 'days_since_hospital', 'days_until_hospital'
]

target_col = 'label_fall_within7'

#drop rows with missing features or labels
# Only drop rows where label is missing
df = df.dropna(subset=[target_col])

#split into X and y
X = df[features].values
y = df[target_col].astype(int).values
groups = df['subid'].values

#use GroupShuffleSplit to split the data into train and test sets no data leakage
#80% train, 20% test
splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
train_idx, test_idx = next(splitter.split(X, y, groups=groups))

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

#log to wandb
wandb.init(project="detect-xgboost", 
		name="xgb_fall_within7",
		config={
			"imputer": "vae",
			"model": "XGboost",
			"label_shift_days": 7,
			"features": features,
		}
)

from wandb.integration.xgboost import WandbCallback
callbacks = [WandbCallback()]

#verify the split
train_subids = set(groups[train_idx])
test_subids = set(groups[test_idx])
overlap = train_subids.intersection(test_subids)
if overlap:
	print(f"Overlap between train and test sets: {overlap}")
else:
	print("No overlap between train and test sets.")
 
# #tran XGBoost Model
# model = XGBClassifier(
# 	use_label_encoder=False,
# 	eval_metric='logloss',
# 	learning_rate=0.1,
# 	random_state=0
# )
# model.fit(X_train, y_train)

dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=features)
dtest = xgb.DMatrix(X_test, label=y_test, feature_names=features)

params = {
	'objective': 'binary:logistic',
	'eval_metric': ['logloss', "auc"],
	"random_state": 0,
}

bst = xgb.train(
	params,
	dtrain=dtrain,
	num_boost_round=200,
	evals=[(dtrain, "train"), (dtest, "eval")],
	callbacks=callbacks
)

#evaluate and print metrics
y_pred = (bst.predict(dtest) > 0.5).astype(int)
y_pred_proba = bst.predict(dtest)

metrics = {
	'accuracy': accuracy_score(y_test, y_pred),
	'precision': precision_score(y_test, y_pred),
	'recall': recall_score(y_test, y_pred),
	'f1': f1_score(y_test, y_pred),
	'roc_auc': roc_auc_score(y_test, y_pred_proba)
}

#compute confusion matrix
cm = confusion_matrix(y_test, y_pred)
labels = ["No Fall", "Fall"]

#print it
print("Confusion Matrix:")
print(cm)

#plot and log to wandb
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
wandb.log({"confusion_matrix": wandb.Image(plt)})
plt.close()

# Build a DataFrame with subid, date, label, prediction
results_df = pd.DataFrame({
    "subid": df.iloc[test_idx]["subid"].values,
    "date": df.iloc[test_idx]["date"].values,
    "true_label": y_test,
    "predicted_label": y_pred,
    "predicted_prob": y_pred_proba
})

# Save locally
results_path = "/mnt/d/DETECT/OUTPUT/xg_boost/xgboost_predictions.csv"
os.makedirs(os.path.dirname(results_path), exist_ok=True)  # Create folder if missing
results_df.to_csv(results_path, index=False)
print(f"Saved predictions to: {results_path}")

wandb.log(metrics)
wandb.finish()