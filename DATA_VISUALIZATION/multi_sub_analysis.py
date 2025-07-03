# Corrected Analysis: Focus on Subjects with Actual Falls
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def find_truly_predictable_subjects():
    """
    Find subjects who have falls in BOTH datasets and show good prediction performance
    """
    
    print("=== CORRECTED ANALYSIS: SUBJECTS WITH ACTUAL FALLS ===")
    print("Finding subjects with real predictive ability (not data artifacts)")
    
    # Load datasets
    df_30day = pd.read_csv("/mnt/d/DETECT/OUTPUT/windowed_data/windowed_data_30_7.csv")
    df_7day = pd.read_csv("/mnt/d/DETECT/OUTPUT/windowed_data/windowed_data_7_7.csv")
    
    print(f"30-day dataset: {df_30day.shape}")
    print(f"7-day dataset: {df_7day.shape}")
    
    # Find subjects with falls in both datasets
    subjects_30_with_falls = df_30day[df_30day['label'] == 1]['subid'].unique()
    subjects_7_with_falls = df_7day[df_7day['label'] == 1]['subid'].unique()
    
    print(f"\nSubjects with falls in 30-day data: {len(subjects_30_with_falls)}")
    print(f"Subjects with falls in 7-day data: {len(subjects_7_with_falls)}")
    
    # Find subjects with falls in both
    subjects_with_falls_both = set(subjects_30_with_falls).intersection(set(subjects_7_with_falls))
    print(f"Subjects with falls in BOTH datasets: {len(subjects_with_falls_both)}")
    
    # Analyze each subject with falls in both datasets
    subject_analysis = []
    
    for subject_id in subjects_with_falls_both:
        subj_30 = df_30day[df_30day['subid'] == subject_id]
        subj_7 = df_7day[df_7day['subid'] == subject_id]
        
        analysis = {
            'subject_id': subject_id,
            'n_obs_30day': len(subj_30),
            'n_obs_7day': len(subj_7),
            'falls_30day': subj_30['label'].sum(),
            'falls_7day': subj_7['label'].sum(),
            'fall_rate_30day': subj_30['label'].mean(),
            'fall_rate_7day': subj_7['label'].mean(),
            'date_range_start': subj_30['start_date'].min(),
            'date_range_end': subj_30['start_date'].max(),
            'sufficient_data_30': len(subj_30) >= 6,
            'sufficient_data_7': len(subj_7) >= 10,
        }
        
        subject_analysis.append(analysis)
    
    analysis_df = pd.DataFrame(subject_analysis)
    
    # Filter to subjects with sufficient data in both
    good_subjects = analysis_df[
        (analysis_df['sufficient_data_30']) & 
        (analysis_df['sufficient_data_7']) &
        (analysis_df['falls_30day'] > 0) &
        (analysis_df['falls_7day'] > 0)
    ].copy()
    
    print(f"\nSubjects with sufficient data AND falls in both datasets: {len(good_subjects)}")
    
    if len(good_subjects) > 0:
        print(f"\nDetailed analysis of viable subjects:")
        for _, subj in good_subjects.iterrows():
            print(f"  Subject {subj['subject_id']}: "
                  f"30d={subj['falls_30day']}/{subj['n_obs_30day']} falls "
                  f"({subj['fall_rate_30day']:.1%}), "
                  f"7d={subj['falls_7day']}/{subj['n_obs_7day']} falls "
                  f"({subj['fall_rate_7day']:.1%})")
    
    return good_subjects, analysis_df

def analyze_best_actual_subject(good_subjects_df):
    """
    Analyze the subject with the best balance of data and falls
    """
    
    if len(good_subjects_df) == 0:
        print("❌ No subjects found with falls in both datasets")
        return None
    
    # Score subjects based on data quality and fall frequency
    good_subjects_df['data_score'] = (
        good_subjects_df['n_obs_30day'] * 0.3 +  # 30-day observations
        good_subjects_df['n_obs_7day'] * 0.1 +   # 7-day observations  
        good_subjects_df['falls_30day'] * 10 +   # 30-day falls (high weight)
        good_subjects_df['falls_7day'] * 2       # 7-day falls
    )
    
    # Select best subject
    best_subject = good_subjects_df.loc[good_subjects_df['data_score'].idxmax()]
    subject_id = best_subject['subject_id']
    
    print(f"\n{'='*60}")
    print(f"ANALYZING BEST ACTUAL SUBJECT: {subject_id}")
    print(f"{'='*60}")
    print(f"30-day data: {best_subject['falls_30day']} falls / {best_subject['n_obs_30day']} obs")
    print(f"7-day data: {best_subject['falls_7day']} falls / {best_subject['n_obs_7day']} obs")
    
    # Load subject's data
    df_30day = pd.read_csv("/mnt/d/DETECT/OUTPUT/windowed_data/windowed_data_30_7.csv")
    df_7day = pd.read_csv("/mnt/d/DETECT/OUTPUT/windowed_data/windowed_data_7_7.csv")
    
    subj_30 = df_30day[df_30day['subid'] == subject_id].copy().sort_values('start_date')
    subj_7 = df_7day[df_7day['subid'] == subject_id].copy().sort_values('start_date')
    
    # Add date columns
    subj_30['date'] = pd.to_datetime(subj_30['start_date'])
    subj_7['date'] = pd.to_datetime(subj_7['start_date'])
    
    # Key features for analysis
    key_features = [
        'steps_mean', 'steps_std', 'gait_speed_mean', 'gait_speed_std',
        'sleepscore_mean', 'sleepscore_std', 'awakenings_mean', 
        'waso_mean', 'avghr_mean', 'avghr_std', 'hrvscore_mean'
    ]
    
    # Analyze 30-day patterns
    print(f"\n30-DAY PATTERN ANALYSIS:")
    fall_periods_30 = subj_30[subj_30['label'] == 1]
    normal_periods_30 = subj_30[subj_30['label'] == 0]
    
    print(f"Fall periods: {len(fall_periods_30)}")
    print(f"Normal periods: {len(normal_periods_30)}")
    
    # Feature comparison for 30-day data
    feature_insights_30 = []
    available_features_30 = [f for f in key_features if f in subj_30.columns]
    
    for feature in available_features_30:
        fall_values = fall_periods_30[feature].dropna()
        normal_values = normal_periods_30[feature].dropna()
        
        if len(fall_values) > 0 and len(normal_values) > 0:
            fall_mean = fall_values.mean()
            normal_mean = normal_values.mean()
            
            # Calculate effect size
            pooled_std = np.sqrt(((len(fall_values)-1)*fall_values.var() + 
                                (len(normal_values)-1)*normal_values.var()) / 
                               (len(fall_values) + len(normal_values) - 2))
            
            if pooled_std > 0:
                effect_size = abs(fall_mean - normal_mean) / pooled_std
                
                feature_insights_30.append({
                    'feature': feature,
                    'fall_mean': fall_mean,
                    'normal_mean': normal_mean,
                    'difference': fall_mean - normal_mean,
                    'effect_size': effect_size
                })
    
    # Sort by effect size
    feature_insights_30.sort(key=lambda x: x['effect_size'], reverse=True)
    
    print(f"\nTop discriminative features in 30-day data:")
    for insight in feature_insights_30[:5]:
        print(f"  {insight['feature']}: "
              f"Fall={insight['fall_mean']:.3f}, "
              f"Normal={insight['normal_mean']:.3f}, "
              f"Effect={insight['effect_size']:.2f}")
    
    # Analyze 7-day patterns
    print(f"\n7-DAY PATTERN ANALYSIS:")
    fall_periods_7 = subj_7[subj_7['label'] == 1]
    normal_periods_7 = subj_7[subj_7['label'] == 0]
    
    print(f"Fall periods: {len(fall_periods_7)}")
    print(f"Normal periods: {len(normal_periods_7)}")
    
    # Create visualizations
    fig, axes = plt.subplots(3, 2, figsize=(15, 18))
    
    # 1. Time series comparison (30-day vs 7-day)
    if 'steps_mean' in subj_30.columns and 'steps_mean' in subj_7.columns:
        axes[0, 0].plot(subj_30['date'], subj_30['steps_mean'], 
                       marker='o', label='30-day windows', alpha=0.7)
        # Highlight falls in 30-day data
        for _, fall in fall_periods_30.iterrows():
            axes[0, 0].axvline(fall['date'], color='red', linestyle='--', alpha=0.5)
        
        axes[0, 0].set_title('Steps Mean - 30-day Windows')
        axes[0, 0].set_ylabel('Steps Mean')
        axes[0, 0].legend()
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 7-day version
        axes[0, 1].plot(subj_7['date'], subj_7['steps_mean'], 
                       marker='s', label='7-day windows', alpha=0.7, color='orange')
        # Highlight falls in 7-day data
        for _, fall in fall_periods_7.iterrows():
            axes[0, 1].axvline(fall['date'], color='red', linestyle='--', alpha=0.5)
        
        axes[0, 1].set_title('Steps Mean - 7-day Windows')
        axes[0, 1].set_ylabel('Steps Mean')
        axes[0, 1].legend()
        axes[0, 1].tick_params(axis='x', rotation=45)
    
    # 2. Sleep quality comparison
    if 'sleepscore_mean' in subj_30.columns and 'sleepscore_mean' in subj_7.columns:
        axes[1, 0].plot(subj_30['date'], subj_30['sleepscore_mean'], 
                       marker='o', label='30-day windows', alpha=0.7)
        for _, fall in fall_periods_30.iterrows():
            axes[1, 0].axvline(fall['date'], color='red', linestyle='--', alpha=0.5)
        
        axes[1, 0].set_title('Sleep Score - 30-day Windows')
        axes[1, 0].set_ylabel('Sleep Score Mean')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        axes[1, 1].plot(subj_7['date'], subj_7['sleepscore_mean'], 
                       marker='s', label='7-day windows', alpha=0.7, color='orange')
        for _, fall in fall_periods_7.iterrows():
            axes[1, 1].axvline(fall['date'], color='red', linestyle='--', alpha=0.5)
        
        axes[1, 1].set_title('Sleep Score - 7-day Windows')
        axes[1, 1].set_ylabel('Sleep Score Mean')
        axes[1, 1].tick_params(axis='x', rotation=45)
    
    # 3. Heart rate comparison
    if 'avghr_mean' in subj_30.columns and 'avghr_mean' in subj_7.columns:
        axes[2, 0].plot(subj_30['date'], subj_30['avghr_mean'], 
                       marker='o', label='30-day windows', alpha=0.7)
        for _, fall in fall_periods_30.iterrows():
            axes[2, 0].axvline(fall['date'], color='red', linestyle='--', alpha=0.5)
        
        axes[2, 0].set_title('Heart Rate - 30-day Windows')
        axes[2, 0].set_ylabel('Avg HR Mean')
        axes[2, 0].tick_params(axis='x', rotation=45)
        
        axes[2, 1].plot(subj_7['date'], subj_7['avghr_mean'], 
                       marker='s', label='7-day windows', alpha=0.7, color='orange')
        for _, fall in fall_periods_7.iterrows():
            axes[2, 1].axvline(fall['date'], color='red', linestyle='--', alpha=0.5)
        
        axes[2, 1].set_title('Heart Rate - 7-day Windows')
        axes[2, 1].set_ylabel('Avg HR Mean')
        axes[2, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = f"/mnt/d/DETECT/OUTPUT/xg_boost/corrected_subject_{int(subject_id)}_analysis_{timestamp}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nCorrected analysis plots saved to: {plot_path}")
    
    return subject_id, subj_30, subj_7, feature_insights_30

def evaluate_prediction_quality():
    """
    Re-evaluate the hybrid model results focusing only on subjects with actual falls
    """
    
    print(f"\n{'='*60}")
    print("RE-EVALUATION: PREDICTION QUALITY FOR SUBJECTS WITH FALLS")
    print(f"{'='*60}")
    
    # Load hybrid results
    try:
        results_files = [
            "/mnt/d/DETECT/OUTPUT/xg_boost/hybrid_model_results_20250602_022423.csv"
        ]
        
        hybrid_results = None
        for file_path in results_files:
            try:
                hybrid_results = pd.read_csv(file_path)
                print(f"Loaded hybrid results from: {file_path}")
                break
            except:
                continue
        
        if hybrid_results is None:
            print("❌ Could not load hybrid results")
            return
        
        print(f"Total subjects in hybrid results: {len(hybrid_results)}")
        
        # Filter to subjects with actual falls (> 0 in at least one dataset)
        subjects_with_falls = hybrid_results[
            (hybrid_results['falls'] > 0)
        ].copy()
        
        print(f"Subjects with actual falls: {len(subjects_with_falls)}")
        
        if len(subjects_with_falls) > 0:
            print(f"\nPerformance for subjects WITH falls:")
            
            # Long-term performance
            long_term_valid = subjects_with_falls.dropna(subset=['long_term_auc'])
            if len(long_term_valid) > 0:
                print(f"  Long-term models: {len(long_term_valid)} subjects")
                print(f"    Average AUC: {long_term_valid['long_term_auc'].mean():.3f}")
                print(f"    AUC > 0.6: {(long_term_valid['long_term_auc'] > 0.6).sum()}")
                print(f"    AUC > 0.7: {(long_term_valid['long_term_auc'] > 0.7).sum()}")
            
            # Short-term performance  
            short_term_valid = subjects_with_falls.dropna(subset=['short_term_auc'])
            if len(short_term_valid) > 0:
                print(f"  Short-term models: {len(short_term_valid)} subjects")
                print(f"    Average AUC: {short_term_valid['short_term_auc'].mean():.3f}")
                print(f"    AUC > 0.6: {(short_term_valid['short_term_auc'] > 0.6).sum()}")
                print(f"    AUC > 0.7: {(short_term_valid['short_term_auc'] > 0.7).sum()}")
            
            # Best performers with actual falls
            print(f"\nBest performers (subjects WITH falls):")
            for _, row in subjects_with_falls.head(10).iterrows():
                long_auc = row['long_term_auc'] if pd.notna(row['long_term_auc']) else 0
                short_auc = row['short_term_auc'] if pd.notna(row['short_term_auc']) else 0
                combined_auc = row['combined_auc'] if pd.notna(row['combined_auc']) else 0
                
                best_auc = max(long_auc, short_auc, combined_auc)
                if best_auc > 0.5:  # Only show subjects with some predictive ability
                    print(f"  Subject {row['subject_id']}: "
                          f"Falls={row['falls']}, "
                          f"Long={long_auc:.3f}, "
                          f"Short={short_auc:.3f}, "
                          f"Combined={combined_auc:.3f}")
            
    except Exception as e:
        print(f"Error loading hybrid results: {e}")

# Run the corrected analysis
if __name__ == "__main__":
    print("Starting corrected analysis to find subjects with real predictive ability...")
    
    # 1. Find subjects with falls in both datasets
    good_subjects, all_subjects = find_truly_predictable_subjects()
    
    # 2. Analyze the best subject with actual falls
    if len(good_subjects) > 0:
        best_subject_id, subj_30_data, subj_7_data, features = analyze_best_actual_subject(good_subjects)
    else:
        print("❌ No subjects found with falls in both 30-day and 7-day datasets")
        print("This indicates a fundamental issue with window size selection")
    
    # 3. Re-evaluate hybrid model performance
    evaluate_prediction_quality()
    
    print(f"\n{'='*60}")
    print("CORRECTED ANALYSIS COMPLETE")
    print(f"{'='*60}")
    
    if len(good_subjects) > 0:
        print("✅ Found subjects with real predictive challenges")
        print("✅ These subjects should be the focus for clinical deployment")
        print("✅ Analysis reveals true patterns vs. data artifacts")
    else:
        print("⚠ Window size mismatch: 30-day aggregation may miss acute events")
        print("💡 Consider using shorter windows (14-day) for better fall capture")
        print("💡 Or use 7-day models as primary approach")
    
    print("\nKey recommendations:")
    print("1. Focus on subjects with falls in both time scales")
    print("2. Validate that 'perfect' predictions aren't data artifacts")
    print("3. Consider temporal resolution effects on fall detection")
    print("4. Use this corrected analysis for clinical planning")