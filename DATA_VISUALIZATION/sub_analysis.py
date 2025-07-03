# Subject 1638 Perfect Prediction Analysis
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def analyze_perfect_subject(subject_id=1806.0):
    """
    Deep dive analysis of Subject 1638 who achieved perfect prediction (AUC = 1.0)
    """
    
    print(f"=== DEEP DIVE ANALYSIS: SUBJECT {subject_id} ===")
    print("This subject achieved PERFECT fall prediction (AUC = 1.0)")
    print("Understanding their patterns could unlock better prediction for others")
    
    # Load both datasets
    df_30day = pd.read_csv("/mnt/d/DETECT/OUTPUT/windowed_data/windowed_data_30_7.csv")
    df_7day = pd.read_csv("/mnt/d/DETECT/OUTPUT/windowed_data/windowed_data_7_7.csv")
    
    # Extract subject's data
    subj_30 = df_30day[df_30day['subid'] == subject_id].copy().sort_values('start_date')
    subj_7 = df_7day[df_7day['subid'] == subject_id].copy().sort_values('start_date')
    
    print(f"\nData Summary for Subject {subject_id}:")
    print(f"30-day windows: {len(subj_30)} observations")
    print(f"7-day windows: {len(subj_7)} observations")
    print(f"30-day falls: {subj_30['label'].sum()}")
    print(f"7-day falls: {subj_7['label'].sum()}")
    print(f"Date range: {subj_30['start_date'].min()} to {subj_30['start_date'].max()}")
    
    # Key features for analysis
    key_features = [
        'steps_mean', 'steps_std', 'gait_speed_mean', 'gait_speed_std',
        'sleepscore_mean', 'sleepscore_std', 'awakenings_mean', 
        'waso_mean', 'avghr_mean', 'avghr_std', 'hrvscore_mean'
    ]
    
    # Analyze 30-day patterns (where perfect prediction was achieved)
    print(f"\n{'='*50}")
    print("30-DAY WINDOW ANALYSIS (Perfect Prediction)")
    print(f"{'='*50}")
    
    # Add date for time series analysis
    subj_30['date'] = pd.to_datetime(subj_30['start_date'])
    
    # Compare fall vs non-fall periods
    fall_periods = subj_30[subj_30['label'] == 1]
    normal_periods = subj_30[subj_30['label'] == 0]
    
    print(f"Fall periods: {len(fall_periods)}")
    print(f"Normal periods: {len(normal_periods)}")
    
    # Feature analysis
    print(f"\nFeature Analysis (Fall vs Normal periods):")
    feature_insights = []
    
    for feature in key_features:
        if feature in subj_30.columns:
            fall_values = fall_periods[feature].dropna()
            normal_values = normal_periods[feature].dropna()
            
            if len(fall_values) > 0 and len(normal_values) > 0:
                fall_mean = fall_values.mean()
                normal_mean = normal_values.mean()
                
                # Calculate effect size (standardized difference)
                pooled_std = np.sqrt(((len(fall_values)-1)*fall_values.var() + 
                                    (len(normal_values)-1)*normal_values.var()) / 
                                   (len(fall_values) + len(normal_values) - 2))
                
                if pooled_std > 0:
                    effect_size = abs(fall_mean - normal_mean) / pooled_std
                    
                    feature_insights.append({
                        'feature': feature,
                        'fall_mean': fall_mean,
                        'normal_mean': normal_mean,
                        'difference': fall_mean - normal_mean,
                        'effect_size': effect_size,
                        'discriminative': effect_size > 0.5  # Cohen's medium effect
                    })
    
    # Sort by effect size
    feature_insights = sorted(feature_insights, key=lambda x: x['effect_size'], reverse=True)
    
    print(f"\nTop discriminative features (effect size > 0.5):")
    for insight in feature_insights:
        if insight['discriminative']:
            print(f"  {insight['feature']}: "
                  f"Fall={insight['fall_mean']:.3f}, "
                  f"Normal={insight['normal_mean']:.3f}, "
                  f"Effect={insight['effect_size']:.2f}")
    
    # Temporal pattern analysis
    print(f"\n{'='*50}")
    print("TEMPORAL PATTERN ANALYSIS")
    print(f"{'='*50}")
    
    # Look for trends leading up to falls
    if len(fall_periods) > 0:
        print(f"Fall timing analysis:")
        for _, fall_period in fall_periods.iterrows():
            fall_date = fall_period['date']
            print(f"  Fall period starting: {fall_date.strftime('%Y-%m-%d')}")
            
            # Find preceding normal periods
            preceding = subj_30[subj_30['date'] < fall_date].tail(3)
            if len(preceding) > 0:
                print(f"    Preceding periods: {len(preceding)}")
                
                # Look for warning signs in key features
                for feature in ['steps_mean', 'sleepscore_mean', 'avghr_mean']:
                    if feature in preceding.columns:
                        trend = preceding[feature].values
                        if len(trend) >= 2:
                            recent_change = trend[-1] - trend[-2] if len(trend) >= 2 else 0
                            overall_trend = (trend[-1] - trend[0]) / len(trend) if len(trend) > 1 else 0
                            print(f"      {feature}: recent_change={recent_change:.2f}, "
                                  f"trend={overall_trend:.3f}")
    
    # === CREATE VISUALIZATIONS ===
    fig, axes = plt.subplots(3, 2, figsize=(15, 18))
    
    # 1. Time series of key features
    time_features = ['steps_mean', 'sleepscore_mean', 'avghr_mean']
    colors = ['blue', 'green', 'red']
    
    for i, (feature, color) in enumerate(zip(time_features, colors)):
        if feature in subj_30.columns:
            # Plot time series
            axes[i, 0].plot(subj_30['date'], subj_30[feature], 
                           marker='o', color=color, alpha=0.7)
            
            # Highlight fall periods
            for _, fall_period in fall_periods.iterrows():
                axes[i, 0].axvline(fall_period['date'], color='red', 
                                  linestyle='--', alpha=0.7, label='Fall Period')
            
            axes[i, 0].set_title(f'{feature} Over Time')
            axes[i, 0].set_ylabel(feature)
            axes[i, 0].tick_params(axis='x', rotation=45)
            
            if i == 0:  # Only add legend to first plot
                axes[i, 0].legend()
    
    # 2. Feature distributions (Fall vs Normal)
    for i, feature in enumerate(['steps_mean', 'sleepscore_mean', 'avghr_mean']):
        if feature in subj_30.columns:
            fall_data = fall_periods[feature].dropna()
            normal_data = normal_periods[feature].dropna()
            
            # Create histogram
            axes[i, 1].hist([normal_data, fall_data], bins=10, alpha=0.7,
                           label=['Normal', 'Fall'], density=True)
            axes[i, 1].set_title(f'{feature} Distribution')
            axes[i, 1].set_xlabel(feature)
            axes[i, 1].set_ylabel('Density')
            axes[i, 1].legend()
    
    plt.tight_layout()
    
    # Save the analysis
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = f"/mnt/d/DETECT/OUTPUT/xg_boost/subject_{int(subject_id)}_analysis_{timestamp}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nDetailed plots saved to: {plot_path}")
    
    # === PREDICTIVE PATTERN RECONSTRUCTION ===
    print(f"\n{'='*50}")
    print("PREDICTIVE PATTERN RECONSTRUCTION")
    print(f"{'='*50}")
    
    # Try to understand what made prediction perfect
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    
    # Use available features
    available_features = [f for f in key_features if f in subj_30.columns]
    
    if len(available_features) > 3 and len(subj_30) > 5:
        X = subj_30[available_features].values
        y = subj_30['label'].values
        
        # Preprocess
        imputer = SimpleImputer(strategy='median')
        scaler = StandardScaler()
        
        X_clean = imputer.fit_transform(X)
        X_scaled = scaler.fit_transform(X_clean)
        
        # Fit the model that achieved perfect prediction
        contamination = max(0.05, min(0.3, y.mean() * 2))
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        iso_forest.fit(X_scaled)
        
        # Get anomaly scores
        anomaly_scores = iso_forest.decision_function(X_scaled)
        
        print(f"Anomaly Score Analysis:")
        print(f"Fall periods - Anomaly scores: {anomaly_scores[y==1]}")
        print(f"Normal periods - Anomaly scores: {anomaly_scores[y==0]}")
        
        # Feature importance (approximate)
        feature_importance = []
        for i, feature in enumerate(available_features):
            # Perturb each feature and see impact on anomaly scores
            X_perturbed = X_scaled.copy()
            X_perturbed[:, i] = X_perturbed[:, i] + 0.1  # Small perturbation
            
            perturbed_scores = iso_forest.decision_function(X_perturbed)
            importance = np.mean(np.abs(anomaly_scores - perturbed_scores))
            feature_importance.append((feature, importance))
        
        # Sort by importance
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        print(f"\nApproximate Feature Importance for Perfect Prediction:")
        for feature, importance in feature_importance[:5]:
            print(f"  {feature}: {importance:.4f}")
    
    # === ACTIONABLE INSIGHTS ===
    print(f"\n{'='*50}")
    print("ACTIONABLE CLINICAL INSIGHTS")
    print(f"{'='*50}")
    
    insights = []
    
    # 1. Most discriminative features
    if feature_insights:
        top_feature = feature_insights[0]
        if top_feature['discriminative']:
            direction = "higher" if top_feature['difference'] > 0 else "lower"
            insights.append(f"🎯 KEY PREDICTOR: {top_feature['feature']} is {direction} "
                          f"before falls (effect size: {top_feature['effect_size']:.2f})")
    
    # 2. Temporal patterns
    if len(fall_periods) > 0:
        fall_frequency = len(fall_periods) / len(subj_30)
        insights.append(f"📊 FALL FREQUENCY: {fall_frequency:.1%} of observation periods")
        
        # Look for seasonal patterns
        fall_months = [pd.to_datetime(date).month for date in fall_periods['start_date']]
        if len(set(fall_months)) < len(fall_months):
            insights.append(f"📅 SEASONAL PATTERN: Falls cluster in certain months")
    
    # 3. Monitoring recommendations
    if feature_insights:
        top_3_features = [f['feature'] for f in feature_insights[:3] if f['discriminative']]
        if top_3_features:
            insights.append(f"🔍 MONITOR: Focus on {', '.join(top_3_features)} for early warning")
    
    # 4. Intervention opportunities
    insights.append(f"⚡ INTERVENTION WINDOW: 30-day model provides strategic warning")
    insights.append(f"🎯 SUCCESS FACTOR: This subject's patterns are highly consistent")
    
    print(f"\nKey Insights for Clinical Implementation:")
    for insight in insights:
        print(f"  {insight}")
    
    # === SAVE DETAILED RESULTS ===
    analysis_results = {
        'subject_id': subject_id,
        'total_periods_30day': len(subj_30),
        'fall_periods': len(fall_periods),
        'fall_rate': len(fall_periods) / len(subj_30),
        'date_range_start': subj_30['start_date'].min(),
        'date_range_end': subj_30['start_date'].max(),
        'top_discriminative_features': [f['feature'] for f in feature_insights[:3] if f['discriminative']],
        'key_insights': insights
    }
    
    # Save to CSV
    results_df = pd.DataFrame([analysis_results])
    results_path = f"/mnt/d/DETECT/OUTPUT/xg_boost/subject_{int(subject_id)}_detailed_analysis_{timestamp}.csv"
    results_df.to_csv(results_path, index=False)
    
    # Save feature analysis
    feature_df = pd.DataFrame(feature_insights)
    feature_path = f"/mnt/d/DETECT/OUTPUT/xg_boost/subject_{int(subject_id)}_features_{timestamp}.csv"
    feature_df.to_csv(feature_path, index=False)
    
    print(f"\nDetailed analysis saved:")
    print(f"  Summary: {results_path}")
    print(f"  Features: {feature_path}")
    
    return analysis_results, feature_insights

# Run the analysis
if __name__ == "__main__":
    try:
        results, features = analyze_perfect_subject(1806.0)
        
        print(f"\n{'='*60}")
        print("SUBJECT 1638 ANALYSIS COMPLETE")
        print(f"{'='*60}")
        print("This analysis reveals the patterns behind perfect fall prediction.")
        print("Use these insights to:")
        print("1. Improve models for other subjects")
        print("2. Develop clinical alert criteria")
        print("3. Design targeted interventions")
        print("4. Guide future sensor data collection")
        
    except Exception as e:
        print(f"Error in analysis: {e}")
        print("Please check that the data files exist and subject 1638 is present.")