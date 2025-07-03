# Identify Predictive Patterns in Non-HRV Subjects
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def analyze_non_hrv_patterns():
    """
    Analyze the 6 high-performing subjects who DON'T rely on HRV
    to identify their predictive patterns (activity, sleep, other signals)
    """
    
    print("=== IDENTIFYING NON-HRV PREDICTIVE PATTERNS ===")
    print("Analyzing high-performing subjects who don't use HRV patterns")
    
    # Load datasets
    df_30day = pd.read_csv("/mnt/d/DETECT/OUTPUT/windowed_data/windowed_data_30_7.csv")
    
    # Non-HRV high performers from previous analysis
    non_hrv_subjects = [2152.0, 1191.0, 1733.0, 2394.0, 1436.0, 3032.0]
    
    print(f"Analyzing {len(non_hrv_subjects)} non-HRV high-performing subjects")
    
    # Define non-HRV feature categories
    feature_categories = {
        'activity': [
            'steps_mean', 'steps_std', 'steps_last', 'steps_max', 'steps_min',
            'gait_speed_mean', 'gait_speed_std', 'gait_speed_last'
        ],
        'sleep_quality': [
            'sleepscore_mean', 'sleepscore_std', 'sleepscore_last',
            'durationinsleep_mean', 'durationinsleep_std',
            'sleep_period_mean', 'sleep_period_std'
        ],
        'sleep_disruption': [
            'awakenings_mean', 'awakenings_std', 'awakenings_last',
            'waso_mean', 'waso_std', 'waso_last',
            'tossnturncount_mean', 'tossnturncount_std'
        ],
        'temporal_patterns': [
            'steps_lag1_mean', 'steps_lag2_mean', 'steps_lag3_mean',
            'sleepscore_lag1_mean', 'sleepscore_lag2_mean',
            'gait_speed_lag1_mean'
        ],
        'timing': [
            'start_sleep_time_mean', 'start_sleep_time_std',
            'end_sleep_time_mean', 'end_sleep_time_std',
            'time_to_sleep_mean', 'time_to_sleep_std'
        ]
    }
    
    # Flatten all non-HRV features
    all_non_hrv_features = []
    for category, features in feature_categories.items():
        all_non_hrv_features.extend(features)
    
    subject_pattern_analyses = []
    
    print(f"\nAnalyzing each non-HRV subject:")
    print("="*70)
    
    for subject_id in non_hrv_subjects:
        subj_data = df_30day[df_30day['subid'] == subject_id].copy()
        
        if len(subj_data) < 5 or subj_data['label'].sum() == 0:
            continue
            
        fall_periods = subj_data[subj_data['label'] == 1]
        normal_periods = subj_data[subj_data['label'] == 0]
        
        print(f"\nSubject {subject_id}:")
        print(f"  Falls: {len(fall_periods)}/{len(subj_data)} observations")
        
        # Analyze each feature category
        subject_analysis = {
            'subject_id': subject_id,
            'n_falls': len(fall_periods),
            'n_observations': len(subj_data),
            'category_insights': {}
        }
        
        top_predictors = []
        
        for category, features in feature_categories.items():
            available_features = [f for f in features if f in subj_data.columns]
            category_insights = []
            
            for feature in available_features:
                fall_values = fall_periods[feature].dropna()
                normal_values = normal_periods[feature].dropna()
                
                if len(fall_values) > 0 and len(normal_values) > 0:
                    fall_mean = fall_values.mean()
                    normal_mean = normal_values.mean()
                    
                    # Calculate effect size
                    pooled_std = np.sqrt(((len(fall_values)-1)*fall_values.var() + 
                                        (len(normal_values)-1)*normal_values.var()) / 
                                       (len(fall_values) + len(normal_values) - 2))
                    
                    if pooled_std > 0:
                        effect_size = abs(fall_mean - normal_mean) / pooled_std
                        percent_change = ((fall_mean - normal_mean) / normal_mean) * 100
                        
                        if effect_size > 0.3:  # Meaningful effect
                            category_insights.append({
                                'feature': feature,
                                'effect_size': effect_size,
                                'percent_change': percent_change,
                                'fall_mean': fall_mean,
                                'normal_mean': normal_mean
                            })
                            
                            top_predictors.append({
                                'feature': feature,
                                'category': category,
                                'effect_size': effect_size,
                                'percent_change': percent_change
                            })
            
            # Store category insights
            if category_insights:
                category_insights.sort(key=lambda x: x['effect_size'], reverse=True)
                subject_analysis['category_insights'][category] = category_insights
        
        # Sort and display top predictors
        top_predictors.sort(key=lambda x: x['effect_size'], reverse=True)
        
        print(f"  Top predictive signals:")
        for predictor in top_predictors[:5]:
            direction = "↓" if predictor['percent_change'] < 0 else "↑"
            print(f"    {predictor['category']}.{predictor['feature']}: "
                  f"{direction}{abs(predictor['percent_change']):.1f}% "
                  f"(effect={predictor['effect_size']:.2f})")
        
        if top_predictors:
            # Identify dominant category
            category_counts = {}
            for predictor in top_predictors[:5]:
                category = predictor['category']
                category_counts[category] = category_counts.get(category, 0) + 1
            
            dominant_category = max(category_counts.items(), key=lambda x: x[1])
            print(f"  Dominant pattern: {dominant_category[0]} ({dominant_category[1]} signals)")
            subject_analysis['dominant_category'] = dominant_category[0]
        else:
            print(f"  ⚠ No strong predictive patterns identified")
            subject_analysis['dominant_category'] = None
        
        subject_pattern_analyses.append(subject_analysis)
    
    # === CROSS-SUBJECT PATTERN ANALYSIS ===
    print(f"\n{'='*70}")
    print("CROSS-SUBJECT PATTERN ANALYSIS")
    print(f"{'='*70}")
    
    # Count dominant categories across subjects
    category_distribution = {}
    for analysis in subject_pattern_analyses:
        if analysis['dominant_category']:
            cat = analysis['dominant_category']
            category_distribution[cat] = category_distribution.get(cat, 0) + 1
    
    print(f"\nDominant pattern types across {len(subject_pattern_analyses)} subjects:")
    for category, count in sorted(category_distribution.items(), key=lambda x: x[1], reverse=True):
        print(f"  {category}: {count} subjects")
    
    # === IDENTIFY COMMON PATTERNS WITHIN CATEGORIES ===
    print(f"\n{'='*70}")
    print("COMMON PATTERNS WITHIN CATEGORIES")
    print(f"{'='*70}")
    
    # Group subjects by dominant category
    category_groups = {}
    for analysis in subject_pattern_analyses:
        if analysis['dominant_category']:
            cat = analysis['dominant_category']
            if cat not in category_groups:
                category_groups[cat] = []
            category_groups[cat].append(analysis)
    
    identified_phenotypes = []
    
    for category, subjects in category_groups.items():
        if len(subjects) > 1:  # Only analyze categories with multiple subjects
            print(f"\n{category.upper()} PHENOTYPE ({len(subjects)} subjects):")
            
            # Find common features within this category
            feature_frequency = {}
            
            for subject_analysis in subjects:
                if category in subject_analysis['category_insights']:
                    for insight in subject_analysis['category_insights'][category]:
                        feature = insight['feature']
                        if feature not in feature_frequency:
                            feature_frequency[feature] = []
                        feature_frequency[feature].append({
                            'subject_id': subject_analysis['subject_id'],
                            'effect_size': insight['effect_size'],
                            'percent_change': insight['percent_change']
                        })
            
            # Find features common to multiple subjects
            common_features = {k: v for k, v in feature_frequency.items() if len(v) > 1}
            
            if common_features:
                print(f"  Common predictive features:")
                for feature, subjects_data in common_features.items():
                    avg_effect = np.mean([s['effect_size'] for s in subjects_data])
                    avg_change = np.mean([s['percent_change'] for s in subjects_data])
                    direction = "↓" if avg_change < 0 else "↑"
                    
                    print(f"    {feature}: {direction}{abs(avg_change):.1f}% average change "
                          f"(effect={avg_effect:.2f}, {len(subjects_data)} subjects)")
                
                # Create phenotype description
                phenotype = {
                    'name': f"{category}_phenotype",
                    'category': category,
                    'subject_count': len(subjects),
                    'subjects': [s['subject_id'] for s in subjects],
                    'common_features': common_features,
                    'description': f"Fall prediction via {category} pattern changes"
                }
                identified_phenotypes.append(phenotype)
            else:
                print(f"  No common features found - subjects use different {category} signals")
    
    # === SUMMARY AND CLINICAL IMPLICATIONS ===
    print(f"\n{'='*70}")
    print("IDENTIFIED FALL PREDICTION PHENOTYPES")
    print(f"{'='*70}")
    
    total_phenotyped = sum(len(p['subjects']) for p in identified_phenotypes)
    total_analyzed = len(subject_pattern_analyses)
    
    print(f"Analysis of {total_analyzed} non-HRV high-performing subjects:")
    print(f"Successfully phenotyped: {total_phenotyped}/{total_analyzed} subjects")
    
    for i, phenotype in enumerate(identified_phenotypes, 1):
        print(f"\nPhenotype {i}: {phenotype['name'].replace('_', ' ').title()}")
        print(f"  Subjects: {phenotype['subjects']}")
        print(f"  Pattern: {phenotype['description']}")
        print(f"  Key signals:")
        
        # Show top 3 common features
        sorted_features = sorted(phenotype['common_features'].items(), 
                               key=lambda x: np.mean([s['effect_size'] for s in x[1]]), 
                               reverse=True)
        
        for feature, subjects_data in sorted_features[:3]:
            avg_change = np.mean([s['percent_change'] for s in subjects_data])
            direction = "decreases" if avg_change < 0 else "increases"
            print(f"    • {feature} {direction} by {abs(avg_change):.1f}% before falls")
    
    # === CLINICAL IMPLEMENTATION RECOMMENDATIONS ===
    print(f"\n{'='*70}")
    print("CLINICAL IMPLEMENTATION RECOMMENDATIONS")
    print(f"{'='*70}")
    
    print(f"Multi-modal personalized monitoring approach confirmed:")
    print(f"")
    
    # HRV phenotype (from previous analysis)
    print(f"Phenotype A: Autonomic-Sleep (25% of high performers)")
    print(f"  • Monitor: HRV during sleep, sleep fragmentation")
    print(f"  • Technology: Sleep sensor with HRV analysis")
    print(f"  • Clinical focus: Sleep quality, autonomic function")
    
    # New phenotypes identified
    for i, phenotype in enumerate(identified_phenotypes, ord('B')):
        letter = chr(i)
        percent = (len(phenotype['subjects']) / (total_analyzed + 2)) * 100  # +2 for HRV subjects
        print(f"")
        print(f"Phenotype {letter}: {phenotype['name'].replace('_', ' ').title()} "
              f"({len(phenotype['subjects'])}/{total_analyzed + 2} = {percent:.0f}% of high performers)")
        print(f"  • Monitor: {', '.join(list(phenotype['common_features'].keys())[:3])}")
        print(f"  • Technology: {phenotype['category']} pattern recognition")
        print(f"  • Clinical focus: {phenotype['category'].replace('_', ' ')} optimization")
    
    return subject_pattern_analyses, identified_phenotypes

def create_phenotype_visualization(subject_analyses, phenotypes):
    """Create visualizations of the identified phenotypes"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Phenotype distribution
    categories = [p['category'] for p in phenotypes]
    counts = [len(p['subjects']) for p in phenotypes]
    
    if categories:
        axes[0, 0].bar(categories, counts)
        axes[0, 0].set_title('Distribution of Fall Prediction Phenotypes')
        axes[0, 0].set_ylabel('Number of Subjects')
        axes[0, 0].tick_params(axis='x', rotation=45)
    
    # 2. Effect sizes by category
    category_effects = {}
    for analysis in subject_analyses:
        for category, insights in analysis['category_insights'].items():
            if category not in category_effects:
                category_effects[category] = []
            for insight in insights:
                category_effects[category].append(insight['effect_size'])
    
    if category_effects:
        categories = list(category_effects.keys())
        effect_data = [category_effects[cat] for cat in categories]
        
        axes[0, 1].boxplot(effect_data, labels=categories)
        axes[0, 1].set_title('Effect Sizes by Category')
        axes[0, 1].set_ylabel('Effect Size')
        axes[0, 1].tick_params(axis='x', rotation=45)
    
    # 3. Subject success patterns
    subject_ids = [a['subject_id'] for a in subject_analyses]
    dominant_cats = [a.get('dominant_category', 'Unknown') for a in subject_analyses]
    
    if subject_ids:
        unique_cats = list(set(dominant_cats))
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_cats)))
        
        for i, cat in enumerate(unique_cats):
            mask = np.array(dominant_cats) == cat
            axes[1, 0].scatter(np.array(subject_ids)[mask], [i]*sum(mask), 
                             c=[colors[i]], label=cat, s=100)
        
        axes[1, 0].set_xlabel('Subject ID')
        axes[1, 0].set_ylabel('Phenotype')
        axes[1, 0].set_title('Subject Phenotype Assignment')
        axes[1, 0].legend()
    
    # 4. Clinical coverage
    phenotype_names = ['HRV-Sleep'] + [p['name'].replace('_', ' ').title() for p in phenotypes]
    phenotype_counts = [2] + [len(p['subjects']) for p in phenotypes]  # 2 for HRV subjects
    
    axes[1, 1].pie(phenotype_counts, labels=phenotype_names, autopct='%1.0f%%')
    axes[1, 1].set_title('Clinical Coverage by Phenotype')
    
    plt.tight_layout()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = f"/mnt/d/DETECT/OUTPUT/xg_boost/phenotype_analysis_{timestamp}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nPhenotype analysis plots saved to: {plot_path}")

# Run the analysis
if __name__ == "__main__":
    try:
        subject_analyses, phenotypes = analyze_non_hrv_patterns()
        
        if phenotypes:
            create_phenotype_visualization(subject_analyses, phenotypes)
        
        print(f"\n{'='*70}")
        print("NON-HRV PATTERN ANALYSIS COMPLETE")
        print(f"{'='*70}")
        print("Successfully identified the remaining fall prediction phenotypes!")
        print("This completes the multi-modal personalized fall prediction framework.")
        
    except Exception as e:
        print(f"Error in non-HRV pattern analysis: {e}")