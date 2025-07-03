# Hybrid Multi-Window Fall Prediction Strategy
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class HybridFallDetector:
    """
    Hybrid fall detection using multiple time windows for different prediction horizons
    """
    
    def __init__(self):
        self.long_term_models = {}   # 30-day windows for strategic planning
        self.short_term_models = {}  # 7-day windows for immediate alerts
        self.feature_names = None
        
    def fit_hybrid_model(self, X_long, X_short, y, subject_id, test_size=0.3):
        """
        Fit both long-term and short-term models for a subject
        
        Args:
            X_long: Features from 30-day windows
            X_short: Features from 7-day windows  
            y: Labels (should be aligned)
            subject_id: Subject identifier
        """
        
        results = {
            'subject_id': subject_id,
            'long_term_auc': None,
            'short_term_auc': None,
            'combined_auc': None,
            'n_long': len(X_long),
            'n_short': len(X_short),
            'falls': y.sum()
        }
        
        # Minimum data requirements
        if len(X_long) < 6 or len(X_short) < 10:
            return results
            
        try:
            # === LONG-TERM MODEL (Strategic Risk Assessment) ===
            split_long = int(len(X_long) * (1 - test_size))
            if split_long < 3:
                return results
                
            X_long_train, X_long_test = X_long[:split_long], X_long[split_long:]
            y_long_train, y_long_test = y[:len(X_long)][:split_long], y[:len(X_long)][split_long:]
            
            # Preprocess long-term data
            imputer_long = SimpleImputer(strategy='median')
            scaler_long = RobustScaler()
            
            X_long_clean = imputer_long.fit_transform(X_long_train)
            X_long_scaled = scaler_long.fit_transform(X_long_clean)
            
            # Train long-term model
            contamination_long = max(0.05, min(0.3, y_long_train.mean() * 2))
            model_long = IsolationForest(
                contamination=contamination_long,
                random_state=42,
                n_estimators=200
            )
            model_long.fit(X_long_scaled)
            
            # Test long-term model
            X_long_test_clean = imputer_long.transform(X_long_test)
            X_long_test_scaled = scaler_long.transform(X_long_test_clean)
            scores_long = -model_long.decision_function(X_long_test_scaled)
            
            if len(np.unique(y_long_test)) > 1:
                results['long_term_auc'] = roc_auc_score(y_long_test, scores_long)
            
            # === SHORT-TERM MODEL (Immediate Alerts) ===
            split_short = int(len(X_short) * (1 - test_size))
            if split_short < 5:
                return results
                
            X_short_train, X_short_test = X_short[:split_short], X_short[split_short:]
            y_short_train, y_short_test = y[:len(X_short)][:split_short], y[:len(X_short)][split_short:]
            
            # Preprocess short-term data
            imputer_short = SimpleImputer(strategy='median')
            scaler_short = RobustScaler()
            
            X_short_clean = imputer_short.fit_transform(X_short_train)
            X_short_scaled = scaler_short.fit_transform(X_short_clean)
            
            # Train short-term model
            contamination_short = max(0.05, min(0.3, y_short_train.mean() * 2))
            model_short = IsolationForest(
                contamination=contamination_short,
                random_state=42,
                n_estimators=200
            )
            model_short.fit(X_short_scaled)
            
            # Test short-term model
            X_short_test_clean = imputer_short.transform(X_short_test)
            X_short_test_scaled = scaler_short.transform(X_short_test_clean)
            scores_short = -model_short.decision_function(X_short_test_scaled)
            
            if len(np.unique(y_short_test)) > 1:
                results['short_term_auc'] = roc_auc_score(y_short_test, scores_short)
            
            # === COMBINED MODEL ===
            # For combined evaluation, we need to align the predictions
            # Use the shorter test set length
            min_test_len = min(len(scores_long), len(scores_short))
            if min_test_len > 2:
                # Simple ensemble: weighted average
                # Give more weight to long-term for strategic planning
                combined_scores = (0.7 * scores_long[:min_test_len] + 
                                 0.3 * scores_short[:min_test_len])
                
                y_combined_test = y_long_test[:min_test_len]
                if len(np.unique(y_combined_test)) > 1:
                    results['combined_auc'] = roc_auc_score(y_combined_test, combined_scores)
            
            # Store models for this subject
            self.long_term_models[subject_id] = {
                'model': model_long,
                'imputer': imputer_long,
                'scaler': scaler_long,
                'contamination': contamination_long
            }
            
            self.short_term_models[subject_id] = {
                'model': model_short,
                'imputer': imputer_short,
                'scaler': scaler_short,
                'contamination': contamination_short
            }
            
        except Exception as e:
            print(f"Error processing subject {subject_id}: {e}")
            
        return results

def load_multiple_datasets():
    """Load both 30-day and 7-day windowed datasets"""
    
    # Load 30-day window data
    path_30day = "/mnt/d/DETECT/OUTPUT/windowed_data/windowed_data_30_7.csv"
    df_30day = pd.read_csv(path_30day)
    
    # Load 7-day window data  
    path_7day = "/mnt/d/DETECT/OUTPUT/windowed_data/windowed_data_7_7.csv"
    df_7day = pd.read_csv(path_7day)
    
    print(f"30-day dataset: {df_30day.shape}")
    print(f"7-day dataset: {df_7day.shape}")
    
    return df_30day, df_7day

def align_datasets(df_30day, df_7day, features):
    """
    Align the two datasets by subject and ensure we have corresponding data
    """
    
    # Get subjects present in both datasets
    subjects_30 = set(df_30day['subid'].unique())
    subjects_7 = set(df_7day['subid'].unique())
    common_subjects = subjects_30.intersection(subjects_7)
    
    print(f"Subjects in 30-day data: {len(subjects_30)}")
    print(f"Subjects in 7-day data: {len(subjects_7)}")
    print(f"Common subjects: {len(common_subjects)}")
    
    # Filter to common subjects
    df_30_filtered = df_30day[df_30day['subid'].isin(common_subjects)].copy()
    df_7_filtered = df_7day[df_7day['subid'].isin(common_subjects)].copy()
    
    # Sort by subject and date
    df_30_filtered = df_30_filtered.sort_values(['subid', 'start_date'])
    df_7_filtered = df_7_filtered.sort_values(['subid', 'start_date'])
    
    # Only keep subjects with sufficient data in both datasets
    subject_counts_30 = df_30_filtered['subid'].value_counts()
    subject_counts_7 = df_7_filtered['subid'].value_counts()
    
    sufficient_subjects = set(
        subject_counts_30[subject_counts_30 >= 6].index
    ).intersection(
        set(subject_counts_7[subject_counts_7 >= 10].index)
    )
    
    df_30_final = df_30_filtered[df_30_filtered['subid'].isin(sufficient_subjects)]
    df_7_final = df_7_filtered[df_7_filtered['subid'].isin(sufficient_subjects)]
    
    print(f"Final subjects with sufficient data: {len(sufficient_subjects)}")
    print(f"Final 30-day data: {df_30_final.shape}")
    print(f"Final 7-day data: {df_7_final.shape}")
    
    return df_30_final, df_7_final, sufficient_subjects

# === MAIN ANALYSIS ===
print("=== HYBRID MULTI-WINDOW FALL PREDICTION ===")

# Load datasets
df_30day, df_7day = load_multiple_datasets()

# Define features (use intersection of available features)
base_features = [
    'steps_mean', 'steps_std', 'steps_last', 'gait_speed_mean', 'gait_speed_std',
    'sleepscore_mean', 'sleepscore_std', 'awakenings_mean', 'waso_mean', 'waso_std',
    'avghr_mean', 'avghr_std', 'hrvscore_mean', 'hrvscore_std',
    'bedexitcount_mean', 'tossnturncount_mean',
    'steps_lag1_mean', 'sleepscore_lag1_mean', 'avghr_lag1_mean'
]

# Get features available in both datasets
features_30 = [f for f in base_features if f in df_30day.columns]
features_7 = [f for f in base_features if f in df_7day.columns]
common_features = list(set(features_30).intersection(set(features_7)))

print(f"Using {len(common_features)} common features: {common_features}")

# Align datasets
df_30_aligned, df_7_aligned, valid_subjects = align_datasets(df_30day, df_7day, common_features)

# Initialize hybrid detector
hybrid_detector = HybridFallDetector()

# Evaluate each subject
results = []
print(f"\nEvaluating hybrid models for {len(valid_subjects)} subjects...")

for subject_id in valid_subjects:
    # Get subject data from both datasets
    subj_30 = df_30_aligned[df_30_aligned['subid'] == subject_id]
    subj_7 = df_7_aligned[df_7_aligned['subid'] == subject_id]
    
    if len(subj_30) < 6 or len(subj_7) < 10:
        continue
    
    # Extract features and labels
    X_long = subj_30[common_features].values
    X_short = subj_7[common_features].values
    y_long = subj_30['label'].values
    y_short = subj_7['label'].values
    
    # Use the labels from the dataset with more observations (7-day)
    y_combined = y_short
    
    # Fit hybrid model
    result = hybrid_detector.fit_hybrid_model(X_long, X_short, y_combined, subject_id)
    if result['long_term_auc'] is not None or result['short_term_auc'] is not None:
        results.append(result)

# === ANALYSIS OF RESULTS ===
results_df = pd.DataFrame(results)

print(f"\n{'='*60}")
print("HYBRID MODEL EVALUATION RESULTS")
print(f"{'='*60}")

if len(results_df) > 0:
    print(f"Subjects evaluated: {len(results_df)}")
    
    # Long-term model performance
    long_term_valid = results_df.dropna(subset=['long_term_auc'])
    if len(long_term_valid) > 0:
        print(f"\nLong-term models (30-day windows):")
        print(f"  Subjects with AUC: {len(long_term_valid)}")
        print(f"  Average AUC: {long_term_valid['long_term_auc'].mean():.3f}")
        print(f"  AUC > 0.6: {(long_term_valid['long_term_auc'] > 0.6).sum()}")
        print(f"  AUC > 0.7: {(long_term_valid['long_term_auc'] > 0.7).sum()}")
    
    # Short-term model performance
    short_term_valid = results_df.dropna(subset=['short_term_auc'])
    if len(short_term_valid) > 0:
        print(f"\nShort-term models (7-day windows):")
        print(f"  Subjects with AUC: {len(short_term_valid)}")
        print(f"  Average AUC: {short_term_valid['short_term_auc'].mean():.3f}")
        print(f"  AUC > 0.6: {(short_term_valid['short_term_auc'] > 0.6).sum()}")
        print(f"  AUC > 0.7: {(short_term_valid['short_term_auc'] > 0.7).sum()}")
    
    # Combined model performance
    combined_valid = results_df.dropna(subset=['combined_auc'])
    if len(combined_valid) > 0:
        print(f"\nCombined models:")
        print(f"  Subjects with AUC: {len(combined_valid)}")
        print(f"  Average AUC: {combined_valid['combined_auc'].mean():.3f}")
        print(f"  AUC > 0.6: {(combined_valid['combined_auc'] > 0.6).sum()}")
        print(f"  AUC > 0.7: {(combined_valid['combined_auc'] > 0.7).sum()}")
    
    # Best performing subjects across all models
    print(f"\nTop performing subjects:")
    for _, row in results_df.head(10).iterrows():
        long_auc = row['long_term_auc'] if pd.notna(row['long_term_auc']) else 0
        short_auc = row['short_term_auc'] if pd.notna(row['short_term_auc']) else 0
        combined_auc = row['combined_auc'] if pd.notna(row['combined_auc']) else 0
        
        best_auc = max(long_auc, short_auc, combined_auc)
        if best_auc > 0.6:
            print(f"  Subject {row['subject_id']}: Long={long_auc:.3f}, "
                  f"Short={short_auc:.3f}, Combined={combined_auc:.3f}")

else:
    print("No valid results obtained")

# === SAVE RESULTS ===
if len(results_df) > 0:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = f"/mnt/d/DETECT/OUTPUT/xg_boost/hybrid_model_results_{timestamp}.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\nHybrid model results saved to: {results_path}")

# === CREATE COMPARISON VISUALIZATION ===
if len(results_df) > 0:
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # AUC comparison
    auc_data = []
    for _, row in results_df.iterrows():
        if pd.notna(row['long_term_auc']):
            auc_data.append({'Model': 'Long-term (30d)', 'AUC': row['long_term_auc']})
        if pd.notna(row['short_term_auc']):
            auc_data.append({'Model': 'Short-term (7d)', 'AUC': row['short_term_auc']})
        if pd.notna(row['combined_auc']):
            auc_data.append({'Model': 'Combined', 'AUC': row['combined_auc']})
    
    if auc_data:
        auc_df = pd.DataFrame(auc_data)
        auc_df.boxplot(column='AUC', by='Model', ax=axes[0,0])
        axes[0,0].set_title('AUC Distribution by Model Type')
        axes[0,0].set_ylabel('AUC')
    
    # Performance correlation
    correlation_data = results_df.dropna(subset=['long_term_auc', 'short_term_auc'])
    if len(correlation_data) > 0:
        axes[0,1].scatter(correlation_data['long_term_auc'], correlation_data['short_term_auc'])
        axes[0,1].set_xlabel('Long-term AUC')
        axes[0,1].set_ylabel('Short-term AUC')
        axes[0,1].set_title('Long-term vs Short-term Performance')
        
        # Add correlation coefficient
        corr = correlation_data['long_term_auc'].corr(correlation_data['short_term_auc'])
        axes[0,1].text(0.05, 0.95, f'Correlation: {corr:.3f}', 
                      transform=axes[0,1].transAxes)
    
    # Data size vs performance
    axes[1,0].scatter(results_df['n_long'], results_df['long_term_auc'], 
                     alpha=0.6, label='Long-term')
    axes[1,0].scatter(results_df['n_short'], results_df['short_term_auc'], 
                     alpha=0.6, label='Short-term')
    axes[1,0].set_xlabel('Number of Observations')
    axes[1,0].set_ylabel('AUC')
    axes[1,0].set_title('Data Amount vs Performance')
    axes[1,0].legend()
    
    # Success rate by model type
    success_rates = []
    if len(long_term_valid) > 0:
        success_rates.append({
            'Model': 'Long-term', 
            'Success Rate': (long_term_valid['long_term_auc'] > 0.6).mean()
        })
    if len(short_term_valid) > 0:
        success_rates.append({
            'Model': 'Short-term', 
            'Success Rate': (short_term_valid['short_term_auc'] > 0.6).mean()
        })
    if len(combined_valid) > 0:
        success_rates.append({
            'Model': 'Combined', 
            'Success Rate': (combined_valid['combined_auc'] > 0.6).mean()
        })
    
    if success_rates:
        success_df = pd.DataFrame(success_rates)
        axes[1,1].bar(success_df['Model'], success_df['Success Rate'])
        axes[1,1].set_ylabel('Success Rate (AUC > 0.6)')
        axes[1,1].set_title('Success Rate by Model Type')
        axes[1,1].set_ylim(0, 1)
    
    plt.tight_layout()
    
    plot_path = f"/mnt/d/DETECT/OUTPUT/xg_boost/hybrid_model_comparison_{timestamp}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Comparison plots saved to: {plot_path}")

print(f"\n{'='*60}")
print("HYBRID MODEL RECOMMENDATIONS")
print(f"{'='*60}")
print("✓ Use 30-day windows for strategic risk assessment")
print("✓ Use 7-day windows for immediate alert generation")  
print("✓ Combine both for comprehensive fall prevention")
print("✓ Focus on subjects showing good performance in either model")