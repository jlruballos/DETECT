import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

base_path = '/mnt/d/DETECT'

# Create output directory if it does not exist
output_path = os.path.join(base_path, 'OUTPUT', 'KMeans_Clustering_Yearly')
os.makedirs(output_path, exist_ok=True)
plot_path = os.path.join(output_path, "plots")
os.makedirs(plot_path, exist_ok=True)
box_plot_path = os.path.join(output_path, "boxplots")
os.makedirs(box_plot_path, exist_ok=True)

#df = pd.read_csv("/mnt/d/DETECT/OUTPUT/sequence_feature/labeled_daily_data_mean.csv", parse_dates=["date"])
df = pd.read_csv("/mnt/d/DETECT/OUTPUT/impute_data/labeled_daily_data_mean_imputed.csv", parse_dates=["date"])


demo_cols = ['birthyr', 'sex', 'hispanic', 'race', 'educ', 'livsitua', 'independ', 'residenc', 'alzdis', 'maristat','moca_avg', 'cogstat', 'amyloid']


df_clean = df.copy()

#intial amyloid participants
initial_amyloid_pos = set(df_clean[df_clean['amyloid'] == 1]['subid'].unique())

# Example preprocessing
demo_df = df_clean[demo_cols].copy()

# Drop amyloid for unsupervised clustering
demo_df = demo_df.drop(columns=['amyloid'])

#create a new column for age from birth year
df_clean['age'] = df_clean['date'].dt.year - df_clean['birthyr']

#encode sex as numeric
df_clean['sex_encoded'] = df_clean['sex'].map({1: 1, 2: 0})

#encode cognitive status as numeric
cogstat_map = {
    1: 0,  # Better than normal
    2: 1,  # Normal
    3: 2,  # One or two abnormal scores
    4: 3,  # Three or more abnormal scores
    0: -1  # Unable to render opinion
}

df_clean['cogstat_encoded'] = df_clean['cogstat'].map(cogstat_map)

df_clean['hispanic_encoded'] = df_clean['hispanic'].fillna(0).astype(int)
df_clean['race_encoded'] = df_clean['race'].fillna(-1).astype(int)
df_clean['educ_encoded'] = df_clean['educ'].fillna(-1).astype(int)
df_clean['independ_encoded'] = df_clean['independ'].fillna(-1).astype(int)
df_clean['residenc_encoded'] = df_clean['residenc'].fillna(-1).astype(int)
df_clean['livsitua_encoded'] = df_clean['livsitua'].fillna(-1).astype(int)
df_clean['maristat_encoded'] = df_clean['maristat'].fillna(-1).astype(int)
df_clean['moca_avg'] = df_clean['moca_avg'].fillna(df_clean['moca_avg'].mean())

#encode alzheimers disease status
df_clean['alzdis_encoded'] = df_clean['alzdis'].fillna(0).astype(int)  # Fill NaN with 0 and convert to int

# Define feature and demographic columns
feature_cols = [
    'steps', 'awakenings', 'bedexitcount', 'end_sleep_time', 'gait_speed',
     'durationinsleep',  'durationawake', 'sleepscore', #'inbed_time', 'outbed_time',
    'waso', 'hrvscore', 'start_sleep_time', 
     'tossnturncount', 'maxhr', 'avghr', 'avgrr', 'time_in_bed_after_sleep',
      'label_fall', 'label_hospital', 'label_accident', 'label_medication', 'label_mood_lonely', 'label_mood_blue',
      'Night_Bathroom_Visits', 'Night_Kitchen_Visits',
    # #include demographic features
    # 'age', 'sex_encoded', 'cogstat_encoded', 'alzdis_encoded', 'hispanic_encoded',
    # 'race_encoded', 'educ_encoded', 'independ_encoded', 'residenc_encoded',
    # 'livsitua_encoded', 'maristat_encoded', 'moca_avg'
]

#drop rows with NaN in feature columns or date
#df_clean = df_clean.dropna(subset=feature_cols + ['date']).copy()

#fill NaN values in feature columns with the mean of each column
df_clean[feature_cols] = df_clean[feature_cols].fillna(df_clean[feature_cols].mean())

#remove rows with negative values in features
df_clean = df_clean[(df_clean[feature_cols] >= 0).all(axis=1)].copy()


#prepare for clustering
#df_clean['amyloid'] = df_clean['amyloid'].astype(int) # #0 for negative, 1 for positive

df_clean['amyloid'] = df_clean['amyloid'].fillna(-1).astype(int) # -1 for unknown, 0 for negative, 1 for positive

df_clean['year'] = df_clean['date'].dt.year  # Add year column

#cluster per year
yearly_clusters = []
metrics = []

for year, group in df_clean.groupby('year'):
    subgroup = group.dropna(subset=feature_cols).copy()
    if subgroup.empty:
        print(f"Skipping year {year} - no data available.")
        continue
    print(f"Processing year: {year}")

    # Standardize
    X = subgroup[feature_cols]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    #PCA reduction before clustering
    pca = PCA(n_components=0.95)  # Keep 95% variance
    X_pca = pca.fit_transform(X_scaled)
    print(f"Year {year}: Reduced features from {X_scaled.shape[1]} to {X_pca.shape[1]} PCA components.")
    
    # Evaluate feature contributions to retained PCA components
    loading_matrix = pd.DataFrame(
		abs(pca.components_), 
		columns=feature_cols
	)

	# Sum absolute contributions across all retained components
    feature_contributions = loading_matrix.sum(axis=0)

	# Normalize to show percent contribution
    feature_contributions /= feature_contributions.sum()

	# Rank features from least to most contributing
    contribution_ranked = feature_contributions.sort_values()

	# Save full contribution report
    contribution_file = os.path.join(output_path, f"pca_feature_contributions_{year}.csv")
    contribution_ranked.to_frame(name="normalized_contribution").to_csv(contribution_file)
    print(f"Saved PCA feature contributions for {year} to {contribution_file}")

	# Optionally: log the bottom 5 least contributing features
    low_contributors = contribution_ranked.head(5)
    print(f"Year {year} - Least contributing features after PCA:\n{low_contributors}")
    
    # Tune K using both Elbow (Inertia) and Silhouette Score
    inertias = []
    sil_scores = []
    k_range = range(2, 20)  # Adjust range as needed
    best_k, best_sil_score = 2, -1
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(X_scaled)
        inertia = kmeans.inertia_
        sil = silhouette_score(X_scaled, labels)
        
        print(f"Year {year} - k={k} | Inertia: {inertia:.0f} | Silhouette: {sil:.3f}")
        
        #PCA KMeans calculation
        kmeans_pca = KMeans(n_clusters=k, random_state=42)
        labels_pca = kmeans_pca.fit_predict(X_pca)
        inertia_pca = kmeans_pca.inertia_
        sil_pca = silhouette_score(X_pca, labels_pca)
        print(f"Year {year} - k={k} | Inertia (PCA): {inertia_pca:.0f} | Silhouette (PCA): {sil_pca:.3f}")
        
        inertias.append(inertia)
        sil_scores.append(sil)
        
        if sil_pca > best_sil_score:
            best_k, best_sil_score = k, sil_pca

	# Plot Elbow and Silhouette side by side
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

	# Elbow plot
    axs[0].plot(k_range, inertias, marker='o')
    axs[0].set_title(f'Elbow Method (Inertia) - {year}')
    axs[0].set_xlabel('Number of Clusters (k)')
    axs[0].set_ylabel('Inertia')

	# Silhouette plot
    axs[1].plot(k_range, sil_scores, marker='o', color='green')
    axs[1].set_title(f'Silhouette Scores - {year}')
    axs[1].set_xlabel('Number of Clusters (k)')
    axs[1].set_ylabel('Silhouette Score')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_path, f"elbow_silhouette_{year}.png"), dpi=300)
    plt.close()
    
    print(f"Best k for year {year} based on silhouette: {best_k}")

    # Final clustering
    kmeans = KMeans(n_clusters=best_k, random_state=42)
    subgroup['cluster'] = kmeans.fit_predict(X_pca)
    
    composition = subgroup.groupby('cluster')['amyloid'].value_counts(normalize=True).unstack().fillna(0)
    print(f"Amyloid composition in clusters for {year}:\n{composition}")
    composition_file = os.path.join(output_path, f"amyloid_composition_year_{year}.csv")
    composition.to_csv(composition_file)
    print(f"Saved amyloid composition to {composition_file}")

	# Also run 2D PCA for plotting
    subgroup['pca1'], subgroup['pca2'] = X_pca[:, 0], X_pca[:, 1]
    
    #unique participants in each cluster
    participant_counts = subgroup.groupby('cluster')['subid'].nunique()

    # Save metrics
    sil_score = silhouette_score(X_pca, subgroup['cluster']) if best_k > 1 else None
    ch_score = calinski_harabasz_score(X_pca, subgroup['cluster']) if best_k > 1 else None
    db_score = davies_bouldin_score(X_pca, subgroup['cluster']) if best_k > 1 else None
    metrics.append({
        'year': str(year),
        'n_clusters': best_k,
        'silhouette_score': round(sil_score, 4) if sil_score else None,
        'calinski_harabasz_score': round(ch_score, 4) if ch_score else None,
		'davies_bouldin_score': round(db_score, 4) if db_score else None,
		'n_participants': subgroup['subid'].nunique(),
        'min_participants_in_cluster': participant_counts.min(),
        'max_participants_in_cluster': participant_counts.max(),
        'total_days': len(subgroup),
        'unique_participants': subgroup['subid'].nunique(),
    })
    
    plt.figure(figsize=(10, 7))
    sns.scatterplot(
		data=subgroup,
		x='pca1',
		y='pca2',
		hue='cluster',
		palette='Set2',
		s=60,
		legend='full'
	)
    plt.title(f"PCA Clusters - Year: {year}")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

	# Save plot
    plot_filename = f"pca_clusters_year_{str(year)}.png"
    plot_file_path = os.path.join(plot_path, plot_filename)
    plt.savefig(plot_file_path, dpi=300)
    plt.close()
    
    print(f"Saved PCA plot with subid labels to {plot_file_path}")
    
    subgroup['amyloid_status'] = subgroup['amyloid']

	#save filtered participants with clusters
    participant_clusters = (
		subgroup
		.groupby(['subid', 'year', 'amyloid_status', 'cluster'])
		.size()
		.reset_index(name='days')
	)
	# Save to file
    filename = f'participants_year_{year}.csv'
    participant_clusters.to_csv(os.path.join(output_path, filename), index=False)
    print(f"Saved participant info to {filename}")

	#append to yearly clusters
    yearly_clusters.append(subgroup)

# Combine and save
final_df = pd.concat(yearly_clusters, ignore_index=True)
final_df.to_csv(os.path.join(output_path, 'yearly_kmeans_clusters_all.csv'), index=False)
pd.DataFrame(metrics).to_csv(os.path.join(output_path, 'clustering_metrics_all.csv'), index=False)

composition.to_csv(os.path.join(output_path, f"amyloid_composition_year_{year}.csv"))

#feature averages by cluster and year
cluster_summary = final_df.groupby(['year', 'amyloid_status','cluster'])[feature_cols].mean().round(2)
print(cluster_summary)

#save cluster summary to CSV
summary_file = os.path.join(output_path, 'yearly_cluster_summary_amyloid.csv')
cluster_summary.to_csv(summary_file)
print(f"Cluster summary saved to {summary_file}")

#feature averages by cluster and year
cluster_summary = final_df.groupby(['year', 'cluster'])[feature_cols].mean().round(2)

general_summary_file = os.path.join(output_path, 'yearly_cluster_summary.csv')
cluster_summary.to_csv(general_summary_file)
print(f"General cluster summary saved to {general_summary_file}")

# Boxplots
for year in final_df['year'].unique():
    year_data = final_df[final_df['year'] == year]
    for feature in feature_cols:
        plt.figure(figsize=(12, 6))
        ax = sns.boxplot(
            data=year_data,
            x='cluster',
            y=feature,
            palette='Set2',
            hue='cluster',
            showmeans=False,
            legend=False
        )
        clusters = sorted(year_data['cluster'].unique())
        for i, cluster in enumerate(clusters):
            sub_data = year_data[year_data['cluster'] == cluster]
            count = sub_data['subid'].nunique()
            median_val = sub_data[feature].median()
            y_max = sub_data[feature].max()
            ax.text(i, y_max * 1.02, f"n={count}", ha='center', fontsize=9, color='blue')
            ax.text(i, y_max * 0.95, f"med={median_val:.1f}", ha='center', fontsize=9, color='black')

        plt.title(f"{feature} by Cluster - Year {year}")
        plt.xlabel("Cluster")
        plt.ylabel(feature)
        plt.tight_layout()
        filename = f"boxplot_{feature}_year_{year}.png"
        plt.savefig(os.path.join(box_plot_path, filename), dpi=300)
        plt.close()

# Plot cluster counts by year
plt.figure(figsize=(12, 6))
sns.countplot(data=final_df, x='year', hue='cluster', palette='Set2')
plt.title("Cluster Counts by Year")
plt.ylabel("Number of Days")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(plot_path, "cluster_counts_by_year.png"), dpi=300, bbox_inches='tight')
plt.close()  # Close after saving