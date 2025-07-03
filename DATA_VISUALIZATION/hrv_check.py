# HRV Pattern Analysis Across High-Performing Subjects
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

def analyze_hrv_patterns_across_subjects():
    """
    Analyze whether HRV (hrvscore_mean, avghr_mean, avghr_std) is a common 
    predictive factor across all high-performing subjects
    """
    
    print("=== HRV PATTERN ANALYSIS ACROSS HIGH-PERFORMING SUBJECTS ===")
    print("Investigating if HRV is the common factor in successful fall prediction")
    
    # Load datasets
    df_30day = pd.read_csv("/mnt/d/DETECT/OUTPUT/windowed_data/windowed_data_30_7.csv")
    df_7day = pd.read_csv("/mnt/d/DETECT/OUTPUT/windowed_data/windowed_data_7_7.csv")
    
    # Load hybrid results to identify high performers
    try:
        hybrid_results = pd.read_csv("/mnt/d/DETECT/OUTPUT/xg_boost/hybrid_model_results_20250602_022423.csv")
    except:
        print("❌ Could not load hybrid results")
        return
    
    # Define high performers (AUC > 0.6 in any model)
    high_performers = hybrid_results[
        (hybrid_results['long_term_auc'] > 0.6) | 
        (hybrid_results['short_term_auc'] > 0.6) |
        (hybrid_results['combined_auc'] > 0.6)
    ].copy()
    
    print(f"High-performing subjects identified: {len(high_performers)}")
    
    # Focus on subjects with falls in both datasets (genuine predictability)
    subjects_30_with_falls = df_30day[df_30day['label'] == 1]['subid'].unique()
    subjects_7_with_falls = df_7day[df_7day['label'] == 1]['subid'].unique()
    subjects_with_falls_both = set(subjects_30_with_falls).intersection(set(subjects_7_with_falls))
    
    # Filter high performers to those with actual falls
    genuine_high_performers = high_performers[
        high_performers['subject_id'].isin(subjects_with_falls_both)
    ].copy()
    
    print(f"High performers with falls in both datasets: {len(genuine_high_performers)}")
    
    if len(genuine_high_performers) == 0:
        print("❌ No genuine high performers found")
        return
    
    # HRV-related features to analyze
    hrv_features = [
        'hrvscore_mean', 'hrvscore_std', 'hrvscore_last',
        'avghr_mean', 'avghr_std', 'avghr_last',
        'awakenings_mean', 'awakenings_std'  # Sleep-related (autonomic)
    ]
    
    # Analyze each high-performing subject
    subject_analyses = []
    
    print(f"\nAnalyzing HRV patterns for each high-performing subject:")
    print("="*70)
    
    for _, subject_row in genuine_high_performers.iterrows():
        subject_id = subject_row['subject_id']
        
        # Get subject's data from both datasets
        subj_30 = df_30day[df_30day['subid'] == subject_id].copy()
        subj_7 = df_7day[df_7day['subid'] == subject_id].copy()
        
        # Analyze 30-day data (since most high performers use this)
        if len(subj_30) > 5 and subj_30['label'].sum() > 0:
            
            fall_periods = subj_30[subj_30['label'] == 1]
            normal_periods = subj_30[subj_30['label'] == 0]
            
            subject_analysis = {
                'subject_id': subject_id,
                'long_term_auc': subject_row['long_term_auc'],
                'short_term_auc': subject_row['short_term_auc'],
                'combined_auc': subject_row['combined_auc'],
                'n_falls': subj_30['label'].sum(),
                'n_observations': len(subj_30)
            }
            
            # Analyze each HRV feature
            hrv_insights = {}
            for feature in hrv_features:
                if feature in subj_30.columns:
                    fall_values = fall_periods[feature].dropna()
                    normal_values = normal_periods[feature].dropna()
                    
                    if len(fall_values) > 0 and len(normal_values) > 0:
                        fall_mean = fall_values.mean()
                        normal_mean = normal_values.mean()
                        
                        # Calculate effect size (Cohen's d)
                        pooled_std = np.sqrt(((len(fall_values)-1)*fall_values.var() + 
                                            (len(normal_values)-1)*normal_values.var()) / 
                                           (len(fall_values) + len(normal_values) - 2))
                        
                        if pooled_std > 0:
                            effect_size = abs(fall_mean - normal_mean) / pooled_std
                            
                            hrv_insights[feature] = {
                                'fall_mean': fall_mean,
                                'normal_mean': normal_mean,
                                'effect_size': effect_size,
                                'percent_change': ((fall_mean - normal_mean) / normal_mean) * 100,
                                'discriminative': effect_size > 0.5
                            }
            
            subject_analysis['hrv_insights'] = hrv_insights
            subject_analyses.append(subject_analysis)
            
            # Print subject summary
            print(f"\nSubject {subject_id}:")
            print(f"  AUC: Long={subject_row['long_term_auc']:.3f}, "
                  f"Short={subject_row['short_term_auc']:.3f}, "
                  f"Combined={subject_row['combined_auc']:.3f}")
            print(f"  Falls: {subj_30['label'].sum()}/{len(subj_30)} observations")
            
            # Show top HRV features for this subject
            hrv_effects = [(k, v['effect_size']) for k, v in hrv_insights.items() 
                          if v['effect_size'] > 0.3]
            hrv_effects.sort(key=lambda x: x[1], reverse=True)
            
            print(f"  Top HRV predictors:")
            for feature, effect in hrv_effects[:3]:
                insight = hrv_insights[feature]
                direction = "↓" if insight['percent_change'] < 0 else "↑"
                print(f"    {feature}: {direction}{abs(insight['percent_change']):.1f}% "
                      f"(effect={effect:.2f})")
    
    # === CROSS-SUBJECT PATTERN ANALYSIS ===
    print(f"\n{'='*70}")
    print("CROSS-SUBJECT HRV PATTERN ANALYSIS")
    print(f"{'='*70}")
    
    if len(subject_analyses) == 0:
        print("❌ No valid subject analyses")
        return
    
    # Analyze which HRV features are consistently discriminative
    feature_consistency = {}
    
    for feature in hrv_features:
        subjects_with_feature = 0
        discriminative_count = 0
        effect_sizes = []
        percent_changes = []
        
        for analysis in subject_analyses:
            if feature in analysis['hrv_insights']:
                subjects_with_feature += 1
                insight = analysis['hrv_insights'][feature]
                
                if insight['discriminative']:
                    discriminative_count += 1
                
                effect_sizes.append(insight['effect_size'])
                percent_changes.append(insight['percent_change'])
        
        if subjects_with_feature > 0:
            feature_consistency[feature] = {
                'subjects_available': subjects_with_feature,
                'discriminative_count': discriminative_count,
                'discriminative_rate': discriminative_count / subjects_with_feature,
                'avg_effect_size': np.mean(effect_sizes),
                'avg_percent_change': np.mean(percent_changes),
                'consistent_direction': len(set(np.sign(percent_changes))) == 1 if percent_changes else False
            }
    
    # Sort by discriminative rate
    sorted_features = sorted(feature_consistency.items(), 
                           key=lambda x: x[1]['discriminative_rate'], 
                           reverse=True)
    
    print(f"\nHRV Feature Consistency Across {len(subject_analyses)} High Performers:")
    print("-" * 80)
    print(f"{'Feature':<20} {'Subjects':<8} {'Discriminative':<12} {'Rate':<6} {'Avg Effect':<10} {'Avg Change':<10}")
    print("-" * 80)
    
    for feature, stats in sorted_features:
        print(f"{feature:<20} {stats['subjects_available']:<8} "
              f"{stats['discriminative_count']:<12} "
              f"{stats['discriminative_rate']:.1%}   "
              f"{stats['avg_effect_size']:.2f}       "
              f"{stats['avg_percent_change']:+.1f}%")
    
    # === IDENTIFY COMMON HRV PATTERNS ===
    print(f"\n{'='*70}")
    print("COMMON HRV PATTERNS IDENTIFICATION")
    print(f"{'='*70}")
    
    # Features that are discriminative in >50% of subjects
    common_hrv_predictors = [feature for feature, stats in sorted_features 
                           if stats['discriminative_rate'] > 0.5 and stats['subjects_available'] > 2]
    
    if common_hrv_predictors:
        print(f"\n🎯 COMMON HRV PREDICTORS (>50% of subjects):")
        for feature in common_hrv_predictors:
            stats = feature_consistency[feature]
            direction = "decreases" if stats['avg_percent_change'] < 0 else "increases"
            print(f"  ✓ {feature}: {direction} by {abs(stats['avg_percent_change']):.1f}% "
                  f"before falls (effect size: {stats['avg_effect_size']:.2f})")
    else:
        print("❌ No HRV features consistently discriminative across subjects")
    
    # Check if Subject 1806's HRV pattern is representative
    subj_1806_analysis = next((a for a in subject_analyses if a['subject_id'] == 1806), None)
    
    if subj_1806_analysis:
        print(f"\n📊 SUBJECT 1806 HRV PATTERN COMPARISON:")
        print("Is Subject 1806's HRV pattern representative of other high performers?")
        
        if 'hrvscore_mean' in subj_1806_analysis['hrv_insights']:
            s1806_hrv = subj_1806_analysis['hrv_insights']['hrvscore_mean']
            print(f"  Subject 1806 HRV Score: {s1806_hrv['percent_change']:+.1f}% change "
                  f"(effect size: {s1806_hrv['effect_size']:.2f})")
            
            # Compare to other subjects
            other_hrv_changes = []
            for analysis in subject_analyses:
                if (analysis['subject_id'] != 1806 and 
                    'hrvscore_mean' in analysis['hrv_insights']):
                    other_hrv_changes.append(
                        analysis['hrv_insights']['hrvscore_mean']['percent_change']
                    )
            
            if other_hrv_changes:
                avg_other_change = np.mean(other_hrv_changes)
                print(f"  Other subjects HRV Score: {avg_other_change:+.1f}% average change")
                
                if abs(s1806_hrv['percent_change'] - avg_other_change) < 5:
                    print("  ✅ Subject 1806's pattern is REPRESENTATIVE of other high performers")
                else:
                    print("  ⚠ Subject 1806's pattern may be UNIQUE")
    
    # === CREATE VISUALIZATION ===
    if len(subject_analyses) > 1:
        create_hrv_comparison_plots(subject_analyses, feature_consistency, common_hrv_predictors)
    
    # === SUMMARY AND RECOMMENDATIONS ===
    print(f"\n{'='*70}")
    print("SUMMARY: IS HRV THE COMMON FACTOR?")
    print(f"{'='*70}")
    
    total_subjects = len(subject_analyses)
    hrv_subjects = sum(1 for analysis in subject_analyses 
                      if any(insight['discriminative'] for insight in analysis['hrv_insights'].values()))
    
    hrv_rate = hrv_subjects / total_subjects if total_subjects > 0 else 0
    
    print(f"Analysis of {total_subjects} high-performing subjects with actual falls:")
    print(f"")
    print(f"📊 HRV-based prediction success: {hrv_subjects}/{total_subjects} subjects ({hrv_rate:.1%})")
    
    if hrv_rate > 0.7:
        print(f"✅ YES - HRV is the PRIMARY common factor in high-accuracy fall prediction")
        print(f"   Most successful subjects show HRV pattern changes before falls")
    elif hrv_rate > 0.5:
        print(f"🔶 PARTIALLY - HRV is A common factor but not universal")
        print(f"   Some subjects rely on HRV, others use different physiological signals")
    else:
        print(f"❌ NO - HRV is not the primary common factor")
        print(f"   Success appears to come from diverse physiological patterns")
    
    if common_hrv_predictors:
        print(f"\n🎯 Most consistent HRV indicators:")
        for feature in common_hrv_predictors[:3]:
            stats = feature_consistency[feature]
            print(f"   • {feature}: {stats['discriminative_rate']:.0%} of subjects, "
                  f"{stats['avg_percent_change']:+.1f}% average change")
    
    print(f"\n💡 Clinical Implications:")
    if hrv_rate > 0.6:
        print(f"   • HRV monitoring should be PRIMARY approach for fall prediction")
        print(f"   • Sleep-based HRV alerts likely effective for most patients")
        print(f"   • Focus clinical protocols on autonomic/cardiovascular factors")
    else:
        print(f"   • Multi-modal approach needed (HRV + other physiological signals)")
        print(f"   • Subject-specific pattern identification required")
        print(f"   • Cannot rely solely on HRV for all patients")
    
    return subject_analyses, feature_consistency, common_hrv_predictors

def create_hrv_comparison_plots(subject_analyses, feature_consistency, common_predictors):
    """Create visualizations comparing HRV patterns across subjects"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. HRV feature discriminative rates
    features = list(feature_consistency.keys())
    rates = [feature_consistency[f]['discriminative_rate'] for f in features]
    
    axes[0, 0].bar(range(len(features)), rates)
    axes[0, 0].set_xticks(range(len(features)))
    axes[0, 0].set_xticklabels(features, rotation=45, ha='right')
    axes[0, 0].set_ylabel('Discriminative Rate')
    axes[0, 0].set_title('HRV Feature Consistency Across High Performers')
    axes[0, 0].axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='50% threshold')
    axes[0, 0].legend()
    
    # 2. Effect sizes distribution
    if 'hrvscore_mean' in feature_consistency:
        effect_sizes = []
        subject_ids = []
        
        for analysis in subject_analyses:
            if 'hrvscore_mean' in analysis['hrv_insights']:
                effect_sizes.append(analysis['hrv_insights']['hrvscore_mean']['effect_size'])
                subject_ids.append(analysis['subject_id'])
        
        if effect_sizes:
            axes[0, 1].bar(range(len(effect_sizes)), effect_sizes)
            axes[0, 1].set_xticks(range(len(effect_sizes)))
            axes[0, 1].set_xticklabels([f"Subj {int(id)}" for id in subject_ids], rotation=45)
            axes[0, 1].set_ylabel('Effect Size')
            axes[0, 1].set_title('HRV Score Effect Sizes by Subject')
            axes[0, 1].axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Medium effect')
            axes[0, 1].legend()
    
    # 3. Percent changes in HRV
    if 'hrvscore_mean' in feature_consistency:
        percent_changes = []
        for analysis in subject_analyses:
            if 'hrvscore_mean' in analysis['hrv_insights']:
                percent_changes.append(analysis['hrv_insights']['hrvscore_mean']['percent_change'])
        
        if percent_changes:
            axes[1, 0].bar(range(len(percent_changes)), percent_changes)
            axes[1, 0].set_xticks(range(len(percent_changes)))
            axes[1, 0].set_xticklabels([f"Subj {int(id)}" for id in subject_ids], rotation=45)
            axes[1, 0].set_ylabel('Percent Change (%)')
            axes[1, 0].set_title('HRV Score Percent Changes (Fall vs Normal)')
            axes[1, 0].axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    # 4. Performance vs HRV effect size correlation
    performance_scores = []
    hrv_effects = []
    
    for analysis in subject_analyses:
        # Use best AUC as performance measure
        best_auc = max([auc for auc in [analysis['long_term_auc'], 
                                       analysis['short_term_auc'], 
                                       analysis['combined_auc']] 
                       if not pd.isna(auc)])
        
        if 'hrvscore_mean' in analysis['hrv_insights']:
            performance_scores.append(best_auc)
            hrv_effects.append(analysis['hrv_insights']['hrvscore_mean']['effect_size'])
    
    if len(performance_scores) > 1:
        axes[1, 1].scatter(hrv_effects, performance_scores, alpha=0.7)
        axes[1, 1].set_xlabel('HRV Effect Size')
        axes[1, 1].set_ylabel('Best AUC Performance')
        axes[1, 1].set_title('HRV Effect Size vs Prediction Performance')
        
        # Add correlation
        if len(hrv_effects) > 2:
            correlation = np.corrcoef(hrv_effects, performance_scores)[0, 1]
            axes[1, 1].text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                           transform=axes[1, 1].transAxes)
    
    plt.tight_layout()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = f"/mnt/d/DETECT/OUTPUT/xg_boost/hrv_pattern_analysis_{timestamp}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nHRV pattern analysis plots saved to: {plot_path}")

# Run the analysis
if __name__ == "__main__":
    try:
        subject_analyses, feature_consistency, common_predictors = analyze_hrv_patterns_across_subjects()
        
        print(f"\n{'='*70}")
        print("HRV PATTERN ANALYSIS COMPLETE")
        print(f"{'='*70}")
        print("This analysis reveals whether HRV is the universal factor")
        print("in successful fall prediction across high-performing subjects.")
        
    except Exception as e:
        print(f"Error in HRV pattern analysis: {e}")
        import traceback
        traceback.print_exc()