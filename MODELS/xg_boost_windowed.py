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
from datetime import datetime
from sklearn.impute import SimpleImputer

from imblearn.over_sampling import ADASYN, SMOTE
from imblearn.under_sampling import RandomUnderSampler

# Define base paths depending on environment
base_path = '/mnt/d/DETECT'


# Add helpers directory to system path
sys.path.append(os.path.join(base_path, 'HELPERS'))
from helpers import add_future_event_window_labels

csv_path = "/mnt/d/DETECT/OUTPUT/windowed_data/windowed_data_7_7_lag0.csv"
#csv_path = '/mnt/d/DETECT/OUTPUT/raw_export_for_r/imputed_detect_data_pmm_global.csv'
#csv_path = '/mnt/d/DETECT/OUTPUT/raw_export_for_r/raw_daily_data_all_subjects.csv'
df = pd.read_csv(csv_path)

window = 7  # days in the past to consider for features
pred_horizon = 7  # days in the future to predict
lag = 0  # number of lag features to create
# Set target
target_col = 'label'

# Dynamically infer features (excluding ID/date/label columns)
exclude_cols = ['subid', 'start_date', 'end_date', 'label']
features = [col for col in df.columns if col not in exclude_cols]

# Set your imputer name manually or programmatically
imputer = "mean"  # or "vae", "mice", "pmm", etc.

imbalance_strategy = "scale-pos-weight"  # Options: "adasyn", "smote", "none", undersample, scale-pos-weight

# Generate timestamped run name
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
run_name = f"xgb_{imputer}_wind{window}_pred{pred_horizon}_lag{lag}_{imbalance_strategy}_{timestamp}"

print(df.columns)

#add label for "fall within next 7 days"
#df = add_future_event_window_labels(df, event_cols=['label_fall'], horizons=[7])

# features = [
#     'gait_speed', 'steps', 'awakenings', 'bedexitcount', 'end_sleep_time',
#     'inbed_time', 'outbed_time', 'sleepscore', 'durationinsleep', 'durationawake',
#     'waso', 'hrvscore', 'start_sleep_time', 'time_to_sleep', 'time_in_bed_after_sleep',
#     'total_time_in_bed', 'tossnturncount', 'sleep_period', 'minhr', 'maxhr',
#     'avghr', 'avgrr', 'maxrr', 'minrr', 'steps_norm', 'steps_delta',
#     'steps_delta_1d', 'steps_ma_7', 'gait_speed_norm', 'gait_speed_delta',
#     'gait_speed_delta_1d', 'gait_speed_ma_7', 'awakenings_norm', 'awakenings_delta',
#     'awakenings_delta_1d', 'awakenings_ma_7', 'bedexitcount_norm', 'bedexitcount_delta',
#     'bedexitcount_delta_1d', 'bedexitcount_ma_7', 'end_sleep_time_norm',
#     'end_sleep_time_delta', 'end_sleep_time_delta_1d', 'end_sleep_time_ma_7',
#     'inbed_time_norm', 'inbed_time_delta', 'inbed_time_delta_1d', 'inbed_time_ma_7',
#     'outbed_time_norm', 'outbed_time_delta', 'outbed_time_delta_1d', 'outbed_time_ma_7',
#     'sleepscore_norm', 'sleepscore_delta', 'sleepscore_delta_1d', 'sleepscore_ma_7',
#     'durationinsleep_norm', 'durationinsleep_delta', 'durationinsleep_delta_1d',
#     'durationinsleep_ma_7', 'durationawake_norm', 'durationawake_delta',
#     'durationawake_delta_1d', 'durationawake_ma_7', 'waso_norm', 'waso_delta',
#     'waso_delta_1d', 'waso_ma_7', 'hrvscore_norm', 'hrvscore_delta',
#     'hrvscore_delta_1d', 'hrvscore_ma_7', 'start_sleep_time_norm',
#     'start_sleep_time_delta', 'start_sleep_time_delta_1d', 'start_sleep_time_ma_7',
#     'time_to_sleep_norm', 'time_to_sleep_delta', 'time_to_sleep_delta_1d',
#     'time_to_sleep_ma_7', 'time_in_bed_after_sleep_norm', 'time_in_bed_after_sleep_delta',
#     'time_in_bed_after_sleep_delta_1d', 'time_in_bed_after_sleep_ma_7',
#     'total_time_in_bed_norm', 'total_time_in_bed_delta', 'total_time_in_bed_delta_1d',
#     'total_time_in_bed_ma_7', 'tossnturncount_norm', 'tossnturncount_delta',
#     'tossnturncount_delta_1d', 'tossnturncount_ma_7', 'sleep_period_norm',
#     'sleep_period_delta', 'sleep_period_delta_1d', 'sleep_period_ma_7',
#     'minhr_norm', 'minhr_delta', 'minhr_delta_1d', 'minhr_ma_7',
#     'maxhr_norm', 'maxhr_delta', 'maxhr_delta_1d', 'maxhr_ma_7',
#     'avghr_norm', 'avghr_delta', 'avghr_delta_1d', 'avghr_ma_7',
#     'avgrr_norm', 'avgrr_delta', 'avgrr_delta_1d', 'avgrr_ma_7',
#     'maxrr_norm', 'maxrr_delta', 'maxrr_delta_1d', 'maxrr_ma_7',
#     'minrr_norm', 'minrr_delta', 'minrr_delta_1d', 'minrr_ma_7',
#      'days_since_fall' 
# ]

#'days_since_fall', 'days_until_fall', 'days_since_hospital', 'days_until_hospital'

#target_col = 'label_fall_within7'

#drop rows with missing features or labels
# Only drop rows where label is missing
df = df.dropna(subset=[target_col])

print("Class distribution before split:")
print(df[target_col].value_counts())

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

#define scale_pos_weight
scale_pos_weight = None

if imbalance_strategy == "adasyn":
    imputer_obj = SimpleImputer(strategy="mean")
    X_train = imputer_obj.fit_transform(X_train)
    # Apply ADASYN oversampling
    og_training_size = len(y_train)
    adasyn = ADASYN(random_state=0, sampling_strategy='minority', n_neighbors=5)
    X_train, y_train = adasyn.fit_resample(X_train, y_train)
    print(f"Original training size: {og_training_size}, After ADASYN: {len(y_train)}")
    print(f"Class balance after ADASYN: {np.bincount(y_train)}")
elif imbalance_strategy == "undersample":
    imputer_obj = SimpleImputer(strategy="mean")
    X_train = imputer_obj.fit_transform(X_train)
    # Apply RAndomUnderSampler
    og_training_size = len(y_train)
    rus = RandomUnderSampler(random_state=0)
    X_train, y_train = rus.fit_resample(X_train, y_train)
    print(f"Original training size: {og_training_size}, After undersampling: {len(y_train)}")
    print(f"Class balance after ADASYN: {np.bincount(y_train)}")
elif imbalance_strategy == "scale-pos-weight":
    neg, pos = np.bincount(y_train)
    scale_pos_weight = neg / pos
    #scale_pos_weight = 0.7
    print(f"Computed scale_pos_weight: {scale_pos_weight:.2f}")
elif imbalance_strategy == "smote":
    imputer_obj = SimpleImputer(strategy="mean")
    X_train = imputer_obj.fit_transform(X_train)
    og_training_size = len(y_train)
    
    smote = SMOTE(random_state=0, sampling_strategy='minority', k_neighbors=5)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    
    print(f"Original training size: {og_training_size}, After SMOTE: {len(y_train)}")
    print(f"Class balance after SMOTE: {np.bincount(y_train)}")

if imbalance_strategy in ["adasyn", "smote", "undersample"]:
    X_test = imputer_obj.transform(X_test)

#log to wandb
wandb.init(project="detect-windowed-xgboost", 
		name=run_name,
		config={
			"imputer": imputer,
			"imbalance_strategy": imbalance_strategy,
			"model": "XGboost",
			"features": features,
		}
)

# Log class distributions
log_dict= {
    "class_distribution_pre_split": dict(zip(*np.unique(y, return_counts=True))),
    "class_distribution_post_split": dict(zip(*np.unique(y_train, return_counts=True))),
}

if imbalance_strategy == "adasyn":
    log_dict[imbalance_strategy] = len(y_train) - og_training_size
elif imbalance_strategy == "undersample":
	log_dict[imbalance_strategy] = og_training_size - len(y_train)
elif imbalance_strategy == "scale-pos-weight":
	log_dict[imbalance_strategy] = scale_pos_weight
elif imbalance_strategy == "smote":
	log_dict[imbalance_strategy] = len(y_train) - og_training_size

wandb.config.update(log_dict)
    
wandb.log({
    "pre_split_neg_class": log_dict["class_distribution_pre_split"].get(0, 0),
    "pre_split_pos_class": log_dict["class_distribution_pre_split"].get(1, 0),
    "post_split_neg_class": log_dict["class_distribution_post_split"].get(0, 0),
    "post_split_pos_class": log_dict["class_distribution_post_split"].get(1, 0),
    "synthetic_samples": log_dict.get("synthetic_samples", 0),
    "undersampledsamples": log_dict.get("undersampled_samples", 0),
    "scale_pos_weight": log_dict.get("scale_pos_weight", 0)
})

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
if scale_pos_weight:
    params["scale_pos_weight"] = scale_pos_weight
    
bst = xgb.train(
	params,
	dtrain=dtrain,
	num_boost_round=200,
	evals=[(dtrain, "train"), (dtest, "eval")],
	callbacks=callbacks
)
    
# Predict probabilities
y_pred_proba = bst.predict(dtest)

# Threshold sweep
thresholds = np.arange(0.1, 0.91, 0.05)
results = []

for thresh in thresholds:
    y_thresh = (y_pred_proba > thresh).astype(int)
    precision = precision_score(y_test, y_thresh, zero_division=0)
    recall = recall_score(y_test, y_thresh, zero_division=0)
    f1 = f1_score(y_test, y_thresh, zero_division=0)
    results.append({
        "threshold": thresh,
        "precision": precision,
        "recall": recall,
        "f1": f1
    })

# Convert to DataFrame
threshold_df = pd.DataFrame(results)

# Plot threshold tuning curve
plt.figure(figsize=(8, 5))
plt.plot(threshold_df["threshold"], threshold_df["precision"], label="Precision")
plt.plot(threshold_df["threshold"], threshold_df["recall"], label="Recall")
plt.plot(threshold_df["threshold"], threshold_df["f1"], label="F1")
plt.xlabel("Threshold")
plt.ylabel("Score")
plt.title("Threshold Tuning")
plt.legend()
wandb.log({"threshold_tuning_plot": wandb.Image(plt)})
plt.close()

# Select best threshold
best_f1_row = threshold_df.loc[threshold_df["f1"].idxmax()]
best_threshold = best_f1_row["threshold"]
y_pred = (y_pred_proba > best_threshold).astype(int)

# Log best threshold stats
wandb.log({
    "best_threshold": best_threshold,
    "best_f1": best_f1_row["f1"],
    "precision_at_best_f1": best_f1_row["precision"],
    "recall_at_best_f1": best_f1_row["recall"]
})

# Final metrics with best threshold
metrics = {
    'accuracy': accuracy_score(y_test, y_pred),
    'precision': precision_score(y_test, y_pred),
    'recall': recall_score(y_test, y_pred),
    'f1': f1_score(y_test, y_pred),
    'roc_auc': roc_auc_score(y_test, y_pred_proba)
}

wandb.config.update({
    "window_size_days": window,
    "prediction_horizon_days": pred_horizon,
    "lag": lag,
    "num_features": len(features)
})

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


test_df = df.iloc[test_idx].reset_index(drop=True)

# Build a DataFrame with subid, date, label, prediction
results_df = pd.DataFrame({
    "subid": test_df["subid"],
    "start_date": test_df["start_date"],
    "end_date": test_df["end_date"],
    "true_label": y_test,
    "predicted_label": y_pred,
    "predicted_prob": y_pred_proba
})

# Save locally
results_path = f"/mnt/d/DETECT/OUTPUT/xg_boost/xgboost_predictions_{imputer}_{imbalance_strategy}_{timestamp}.csv"
os.makedirs(os.path.dirname(results_path), exist_ok=True)  # Create folder if missing
results_df.to_csv(results_path, index=False)
print(f"Saved predictions to: {results_path}")

wandb.log(metrics)
wandb.finish()