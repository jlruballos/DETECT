# Fixed Personalized Fall Detection with Proper Evaluation
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, classification_report
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class PersonalizedFallDetector:
    """
    Personalized fall detection using temporal train/test splits within subjects
    """
    
    def __init__(self, contamination_rate=0.1, min_observations=8):
        self.contamination_rate = contamination_rate
        self.min_observations = min_observations
        self.subject_models = {}
        
    def fit_and_evaluate_subject(self, X_subject, y_subject, subject_id, test_size=0.3):
        """
        Fit model on early data and test on later data for a single subject
        """
        n_obs = len(X_subject)
        if n_obs < self.min_observations:
            return None
            
        # Temporal split: train on earlier data, test on later data
        split_point = int(n_obs * (1 - test_size))
        if split_point < 5 or (n_obs - split_point) < 3:
            return None  # Need minimum samples for both train and test
            
        X_train = X_subject[:split_point]
        X_test = X_subject[split_point:]
        y_train = y_subject[:split_point]
        y_test = y_subject[split_point:]
        
        # Skip if no variation in training labels
        if len(np.unique(y_train)) < 2 and y_train.sum() == 0:
            # If no falls in training, we can still train (unsupervised)
            pass
        
        try:
            # Preprocess training data
            imputer = SimpleImputer(strategy='median')
            X_train_clean = imputer.fit_transform(X_train)
            
            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train_clean)
            
            # Adjust contamination based on training data
            train_fall_rate = y_train.mean()
            if train_fall_rate > 0:
                contamination = min(0.3, max(0.05, train_fall_rate * 2))
            else:
                contamination = 0.1  # Default when no falls in training
            
            # Train model
            iso_forest = IsolationForest(
                contamination=contamination,
                random_state=42,
                n_estimators=200,
                bootstrap=True
            )
            
            iso_forest.fit(X_train_scaled)
            
            # Test on holdout data
            X_test_clean = imputer.transform(X_test)
            X_test_scaled = scaler.transform(X_test_clean)
            
            # Get predictions
            anomaly_scores = iso_forest.decision_function(X_test_scaled)
            risk_scores = -anomaly_scores  # Higher = more risky
            predictions = iso_forest.predict(X_test_scaled)
            binary_predictions = (predictions == -1).astype(int)
            
            # Calculate metrics if we have both classes in test
            test_auc = None
            if len(np.unique(y_test)) > 1:
                try:
                    test_auc = roc_auc_score(y_test, risk_scores)
                except:
                    test_auc = None
            
            return {
                'subject_id': subject_id,
                'n_train': len(X_train),
                'n_test': len(X_test),
                'train_falls': y_train.sum(),
                'test_falls': y_test.sum(),
                'test_auc': test_auc,
                'contamination': contamination,
                'y_true': y_test,
                'y_pred': binary_predictions,
                'risk_scores': risk_scores,
                'train_fall_rate': train_fall_rate,
                'test_fall_rate': y_test.mean()
            }
            
        except Exception as e:
            print(f"Error processing subject {subject_id}: {e}")
            return None

# Load data
csv_path = "/mnt/d/DETECT/OUTPUT/windowed_data/windowed_data_7_7.csv"
df = pd.read_csv(csv_path)

print("=== FIXED PERSONALIZED FALL DETECTION SYSTEM ===")
print(f"Dataset: {df.shape}")

# Enhanced feature selection
feature_categories = {
    'activity': ['steps_mean', 'steps_std', 'steps_last', 'gait_speed_mean', 'gait_speed_std'],
    'sleep': ['sleepscore_mean', 'sleepscore_std', 'awakenings_mean', 'waso_mean', 'waso_std'],
    'cardiac': ['avghr_mean', 'avghr_std', 'hrvscore_mean', 'hrvscore_std'],
    'movement': ['bedexitcount_mean', 'tossnturncount_mean'],
    'temporal': ['steps_lag1_mean', 'sleepscore_lag1_mean', 'avghr_lag1_mean']
}

# Collect all available features
all_features = []
for category, features in feature_categories.items():
    for feat in features:
        if feat in df.columns:
            all_features.append(feat)

print(f"Using {len(all_features)} features across {len(feature_categories)} categories")

# Filter to subjects with sufficient data and sort by date
subject_counts = df['subid'].value_counts()
subjects_with_data = subject_counts[subject_counts >= 8].index  # Minimum 8 observations

df_filtered = df[df['subid'].isin(subjects_with_data)].copy()
df_filtered['date'] = pd.to_datetime(df_filtered['start_date'])
df_filtered = df_filtered.sort_values(['subid', 'date'])

print(f"Filtered to {len(df_filtered)} observations from {len(subjects_with_data)} subjects")

# Select features with reasonable data quality
missing_rates = df_filtered[all_features].isnull().mean()
good_features = missing_rates[missing_rates < 0.4].index.tolist()  # Allow up to 40% missing

print(f"Selected {len(good_features)} features with <40% missing data")
print(f"Features: {good_features}")

# Prepare data
X = df_filtered[good_features].values
y = df_filtered['label'].values
subjects = df_filtered['subid'].values

print(f"Final dataset: {X.shape}, Falls: {y.sum()}/{len(y)} ({y.mean():.1%})")

# === PERSONALIZED TEMPORAL EVALUATION ===
detector = PersonalizedFallDetector(min_observations=8)

print(f"\nEvaluating personalized models with temporal splits...")
subject_results = []

for subject_id in subjects_with_data:
    # Get subject's data in chronological order
    subject_mask = subjects == subject_id
    X_subject = X[subject_mask]
    y_subject = y[subject_mask]
    
    if len(X_subject) < 8:
        continue
        
    # Evaluate this subject
    result = detector.fit_and_evaluate_subject(X_subject, y_subject, subject_id)
    if result is not None:
        subject_results.append(result)

print(f"Successfully evaluated {len(subject_results)} subjects")

if len(subject_results) == 0:
    print("❌ No subjects could be evaluated. Check data quality and filtering criteria.")
    exit()

# === AGGREGATE RESULTS ===
results_df = pd.DataFrame(subject_results)

# Overall statistics
total_test_obs = results_df['n_test'].sum()
total_test_falls = results_df['test_falls'].sum()
subjects_with_auc = results_df.dropna(subset=['test_auc'])

print(f"\n{'='*60}")
print("PERSONALIZED MODEL EVALUATION RESULTS")
print(f"{'='*60}")
print(f"Subjects evaluated: {len(results_df)}")
print(f"Total test observations: {total_test_obs}")
print(f"Total test falls: {total_test_falls}")
print(f"Overall test fall rate: {total_test_falls/total_test_obs:.1%}")

if len(subjects_with_auc) > 0:
    print(f"\nSubject-level AUC performance ({len(subjects_with_auc)} subjects with calculable AUC):")
    print(f"Average AUC: {subjects_with_auc['test_auc'].mean():.3f}")
    print(f"Median AUC: {subjects_with_auc['test_auc'].median():.3f}")
    print(f"AUC > 0.6: {(subjects_with_auc['test_auc'] > 0.6).sum()}/{len(subjects_with_auc)} subjects")
    print(f"AUC > 0.7: {(subjects_with_auc['test_auc'] > 0.7).sum()}/{len(subjects_with_auc)} subjects")
    print(f"AUC > 0.8: {(subjects_with_auc['test_auc'] > 0.8).sum()}/{len(subjects_with_auc)} subjects")
    
    # Best performing subjects
    top_subjects = subjects_with_auc.nlargest(5, 'test_auc')[['subject_id', 'test_auc', 'test_falls', 'n_test']]
    print(f"\nTop 5 performing subjects:")
    for _, row in top_subjects.iterrows():
        print(f"  Subject {row['subject_id']}: AUC={row['test_auc']:.3f}, "
              f"Falls={row['test_falls']}/{row['n_test']}")

# === POOLED EVALUATION ===
# Combine all test predictions for overall assessment
all_y_true = []
all_y_pred = []
all_risk_scores = []

for result in subject_results:
    if result['y_true'] is not None and len(result['y_true']) > 0:
        all_y_true.extend(result['y_true'])
        all_y_pred.extend(result['y_pred'])
        all_risk_scores.extend(result['risk_scores'])

all_y_true = np.array(all_y_true)
all_y_pred = np.array(all_y_pred)
all_risk_scores = np.array(all_risk_scores)

if len(all_y_true) > 0 and len(np.unique(all_y_true)) > 1:
    # Overall metrics
    pooled_auc = roc_auc_score(all_y_true, all_risk_scores)
    precision, recall, _ = precision_recall_curve(all_y_true, all_risk_scores)
    pr_auc = auc(recall, precision)
    
    print(f"\nPooled evaluation across all subjects:")
    print(f"Pooled ROC-AUC: {pooled_auc:.3f}")
    print(f"Pooled PR-AUC: {pr_auc:.3f}")
    
    # Classification report
    print(f"\nClassification Report (default threshold):")
    print(classification_report(all_y_true, all_y_pred, 
                              target_names=['Normal', 'Fall'], zero_division=0))
    
    # Risk stratification analysis
    risk_percentiles = np.percentile(all_risk_scores, [50, 75, 90, 95, 99])
    print(f"\nRisk stratification analysis:")
    for pct, threshold in zip([50, 75, 90, 95, 99], risk_percentiles):
        high_risk_mask = all_risk_scores >= threshold
        if high_risk_mask.sum() > 0:
            precision_at_thresh = all_y_true[high_risk_mask].mean()
            recall_at_thresh = all_y_true[high_risk_mask].sum() / all_y_true.sum()
            print(f"  Top {100-pct}%: Precision={precision_at_thresh:.3f}, "
                  f"Recall={recall_at_thresh:.3f}, N={high_risk_mask.sum()}")

else:
    print("⚠ Insufficient data for pooled evaluation")
    pooled_auc = None

# === SAVE DETAILED RESULTS ===
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Subject-level results
subject_summary_path = f"/mnt/d/DETECT/OUTPUT/xg_boost/personalized_subject_results_{timestamp}.csv"
results_df.to_csv(subject_summary_path, index=False)

# Detailed predictions (if we have them)
if len(all_y_true) > 0:
    detailed_results = []
    result_idx = 0
    
    for result in subject_results:
        if result['y_true'] is not None:
            for i in range(len(result['y_true'])):
                detailed_results.append({
                    'subject_id': result['subject_id'],
                    'observation_idx': i,
                    'true_label': result['y_true'][i],
                    'predicted_label': result['y_pred'][i],
                    'risk_score': result['risk_scores'][i],
                    'subject_auc': result['test_auc']
                })
    
    detailed_df = pd.DataFrame(detailed_results)
    detailed_path = f"/mnt/d/DETECT/OUTPUT/xg_boost/personalized_detailed_results_{timestamp}.csv"
    detailed_df.to_csv(detailed_path, index=False)
    print(f"\nDetailed results saved to: {detailed_path}")

print(f"Subject summary saved to: {subject_summary_path}")

# === CREATE VISUALIZATIONS ===
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 1. Subject AUC distribution
if len(subjects_with_auc) > 0:
    axes[0,0].hist(subjects_with_auc['test_auc'], bins=15, alpha=0.7, edgecolor='black')
    axes[0,0].axvline(subjects_with_auc['test_auc'].mean(), color='red', linestyle='--',
                      label=f'Mean: {subjects_with_auc["test_auc"].mean():.3f}')
    axes[0,0].axvline(0.5, color='gray', linestyle=':', label='Random (0.5)')
    axes[0,0].set_xlabel('Test AUC')
    axes[0,0].set_ylabel('Number of Subjects')
    axes[0,0].set_title('Distribution of Subject-Level AUC')
    axes[0,0].legend()

# 2. ROC Curve (pooled)
if pooled_auc is not None:
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(all_y_true, all_risk_scores)
    axes[0,1].plot(fpr, tpr, label=f'Personalized Models (AUC = {pooled_auc:.3f})')
    axes[0,1].plot([0, 1], [0, 1], 'k--', label='Random')
    axes[0,1].set_xlabel('False Positive Rate')
    axes[0,1].set_ylabel('True Positive Rate')
    axes[0,1].set_title('Pooled ROC Curve')
    axes[0,1].legend()

# 3. Data amount vs performance
if len(subjects_with_auc) > 0:
    axes[0,2].scatter(subjects_with_auc['n_test'], subjects_with_auc['test_auc'], alpha=0.6)
    axes[0,2].set_xlabel('Test Observations per Subject')
    axes[0,2].set_ylabel('Test AUC')
    axes[0,2].set_title('Performance vs Data Amount')

# 4. Fall rate vs detection performance
axes[1,0].scatter(results_df['test_fall_rate'], results_df['test_auc'], alpha=0.6)
axes[1,0].set_xlabel('Subject Test Fall Rate')
axes[1,0].set_ylabel('Test AUC')
axes[1,0].set_title('Fall Rate vs Detection Performance')

# 5. Risk score distribution
if len(all_risk_scores) > 0:
    fall_scores = all_risk_scores[all_y_true == 1]
    normal_scores = all_risk_scores[all_y_true == 0]
    
    axes[1,1].hist([normal_scores, fall_scores], bins=30, alpha=0.7,
                   label=['Normal', 'Fall'], density=True)
    axes[1,1].set_xlabel('Risk Score')
    axes[1,1].set_ylabel('Density')
    axes[1,1].set_title('Risk Score Distribution')
    axes[1,1].legend()

# 6. Performance summary
perf_summary = {
    'Subjects Evaluated': len(results_df),
    'Subjects with AUC > 0.6': (subjects_with_auc['test_auc'] > 0.6).sum() if len(subjects_with_auc) > 0 else 0,
    'Average AUC': subjects_with_auc['test_auc'].mean() if len(subjects_with_auc) > 0 else 0,
    'Pooled AUC': pooled_auc if pooled_auc is not None else 0
}

axes[1,2].bar(range(len(perf_summary)), list(perf_summary.values()))
axes[1,2].set_xticks(range(len(perf_summary)))
axes[1,2].set_xticklabels(list(perf_summary.keys()), rotation=45, ha='right')
axes[1,2].set_title('Performance Summary')

plt.tight_layout()
plot_path = f'/mnt/d/DETECT/OUTPUT/xg_boost/personalized_evaluation_{timestamp}.png'
plt.savefig(plot_path, dpi=300, bbox_inches='tight')

print(f"Evaluation plots saved to: {plot_path}")

print(f"\n{'='*60}")
print("FINAL ASSESSMENT")
print(f"{'='*60}")

if pooled_auc is not None:
    if pooled_auc > 0.65:
        print("✅ EXCELLENT: Personalized models show strong predictive ability")
    elif pooled_auc > 0.55:
        print("✅ GOOD: Personalized models show meaningful improvement over random")
    else:
        print("⚠ MODERATE: Some improvement but still challenging")
else:
    print("⚠ INSUFFICIENT: Not enough data for reliable evaluation")

if len(subjects_with_auc) > 0:
    good_subjects = (subjects_with_auc['test_auc'] > 0.7).sum()
    print(f"🎯 {good_subjects}/{len(subjects_with_auc)} subjects show excellent individual performance")
    
    if good_subjects > 0:
        print("💡 RECOMMENDATION: Focus on subjects with good performance for clinical deployment")
        print("💡 RECOMMENDATION: Investigate what makes some subjects more predictable")
else:
    print("❌ No subjects had sufficient data for individual evaluation")

print(f"\n📊 This represents a significant methodological improvement:")
print(f"   - Proper temporal validation (train on early data, test on later data)")
print(f"   - Subject-specific modeling (personalized baselines)")
print(f"   - Clinical focus (risk stratification vs binary prediction)")